"""Classify each workflow node and check the observed stage progression.

The evidence judge decides what the transcript supports, including which
assistant actions enter a stage. This module decides, in a fixed order,
whether a node applied, and whether that progression follows the directed graph.
It then scores criterion completion and permitted progression, and assigns a verdict.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from decimal import ROUND_HALF_UP, Decimal
from enum import StrEnum
from types import MappingProxyType

from .inputs import Transcript, Workflow
from .judge import (
    Applicability,
    Criterion,
    CriterionFinding,
    Evidence,
    FindingResult,
    JudgeCode,
    RequirementType,
    Rubric,
    StageEvent,
    Subrequirement,
    fail,
)


class NodeStatus(StrEnum):
    COMPLETED = "completed"
    PARTIAL = "partial"
    INCORRECT = "incorrect"
    SKIPPED = "skipped"
    NOT_APPLICABLE = "not_applicable"
    NOT_REACHED = "not_reached"


class AttemptState(StrEnum):
    ATTEMPTED = "attempted"
    NOT_ATTEMPTED = "not_attempted"


class PathValidity(StrEnum):
    VALID = "valid"
    INVALID = "invalid"
    INCOMPLETE = "incomplete"
    INDETERMINATE = "indeterminate"


class TransitionAssessment(StrEnum):
    PERMITTED = "permitted"
    INVALID = "invalid"
    UNRESOLVED = "unresolved"


class Verdict(StrEnum):
    COMPLETED = "completed"
    PARTIAL = "partial"
    FAILED = "failed"
    INCOMPLETE = "incomplete"
    INDETERMINATE = "indeterminate"


@dataclass(frozen=True, slots=True)
class CriterionAssessment:
    """One top-level criterion after its subrequirements are folded in."""

    criterion_id: str
    applicability: Applicability
    result: FindingResult
    satisfied: bool
    in_denominator: bool
    established: bool
    unresolved: bool
    evidence_turns: tuple[int, ...]
    open_turns: tuple[int, ...]
    missing_ids: tuple[str, ...]
    violated_ids: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class NodeResult:
    node_id: str
    applicability: Applicability
    attempted: AttemptState
    criteria: tuple[CriterionAssessment, ...]
    status: NodeStatus
    evidence_turns: tuple[int, ...]
    missing_requirements: tuple[str, ...]
    violated_requirements: tuple[str, ...]
    satisfied_count: int
    applicable_count: int
    explanation: str


@dataclass(frozen=True, slots=True)
class NodeEvaluation:
    results: tuple[NodeResult, ...]
    by_id: Mapping[str, NodeResult]


@dataclass(frozen=True, slots=True)
class ObservedTransition:
    source: str
    target: str
    assessment: TransitionAssessment


@dataclass(frozen=True, slots=True)
class Traversal:
    """Every start-to-end path, and the one path the established branch selects.

    order_tied is true when two stage actions share a turn and offset.
    """

    candidate_paths: tuple[tuple[str, ...], ...]
    expected_path: tuple[str, ...] | None
    observed_path: tuple[str, ...]
    transitions: tuple[ObservedTransition, ...]
    omitted_nodes: tuple[str, ...]
    unreached_nodes: tuple[str, ...]
    validity: PathValidity
    complete: bool
    order_tied: bool


@dataclass(frozen=True, slots=True)
class NodeScore:
    node_id: str
    score: Decimal | None


@dataclass(frozen=True, slots=True)
class Adherence:
    nodes: tuple[NodeScore, ...]
    node_average: Decimal | None
    transition_score: Decimal
    adherence_score: Decimal | None
    verdict: Verdict


@dataclass(frozen=True, slots=True)
class _Fold:
    observed: tuple[str, ...]
    transitions: tuple[ObservedTransition, ...]
    tied: bool


@dataclass(frozen=True, slots=True)
class _Scope:
    predecessors: Mapping[str, tuple[str, ...]]
    descendants: Mapping[str, frozenset[str]]
    satisfied_nodes: frozenset[str]
    complete_nodes: frozenset[str]
    end_ids: frozenset[str]
    last_user: int | None
    assistant_turns: frozenset[int]


def _require_finding(evidence: Evidence, criterion_id: str) -> CriterionFinding:
    finding = evidence.by_id.get(criterion_id)
    if finding is None:
        fail(JudgeCode.INVALID_EVIDENCE, f"{criterion_id} has no finding.")
    return finding


def _is_met(finding: CriterionFinding) -> bool:
    return (
        finding.applicability is Applicability.APPLICABLE
        and finding.result is FindingResult.MET
    )


def _child_clear(finding: CriterionFinding) -> bool:
    """A not applicable child does not block its parent. Any other open child does."""

    if finding.applicability is Applicability.NOT_APPLICABLE:
        return True
    return _is_met(finding)


def _is_gap(finding: CriterionFinding) -> bool:
    if finding.applicability is not Applicability.APPLICABLE:
        return False
    return finding.result is FindingResult.UNMET or finding.result is FindingResult.INSUFFICIENT


def _is_unresolved_finding(finding: CriterionFinding) -> bool:
    if finding.applicability is Applicability.NOT_APPLICABLE:
        return False
    return (
        finding.applicability is Applicability.UNCERTAIN
        or finding.result is FindingResult.INSUFFICIENT
    )


def _supports(finding: CriterionFinding) -> bool:
    """Met, contradicted, and insufficient citations are execution evidence."""

    if finding.applicability is Applicability.NOT_APPLICABLE:
        return False
    return finding.result is not FindingResult.UNMET


def _is_open(finding: CriterionFinding) -> bool:
    return (
        finding.applicability is not Applicability.NOT_APPLICABLE
        and finding.result is FindingResult.UNMET
    )


def _collect_turns(
    findings: tuple[CriterionFinding, ...],
    include: Callable[[CriterionFinding], bool],
) -> tuple[int, ...]:
    found: set[int] = set()
    for finding in findings:
        if include(finding):
            found.update(finding.evidence_turns)
    return tuple(sorted(found))


def _unique(ids: list[str]) -> tuple[str, ...]:
    seen: set[str] = set()
    ordered: list[str] = []
    for item_id in ids:
        if item_id in seen:
            continue
        seen.add(item_id)
        ordered.append(item_id)
    return tuple(ordered)


def _violated_ids(
    criterion_id: str,
    parent: CriterionFinding,
    pairs: tuple[tuple[Subrequirement, CriterionFinding], ...],
) -> tuple[str, ...]:
    ids: list[str] = []
    if parent.result is FindingResult.CONTRADICTED:
        ids.append(criterion_id)
    for subrequirement, finding in pairs:
        if finding.result is FindingResult.CONTRADICTED:
            ids.append(subrequirement.id)
    return _unique(ids)


def _missing_ids(
    criterion_id: str,
    pairs: tuple[tuple[Subrequirement, CriterionFinding], ...],
    *,
    in_denominator: bool,
    satisfied: bool,
    violated: tuple[str, ...],
) -> tuple[str, ...]:
    if not in_denominator or satisfied:
        return ()
    gaps = [
        subrequirement.id
        for subrequirement, finding in pairs
        if _is_gap(finding) and subrequirement.id not in violated
    ]
    if gaps:
        return tuple(gaps)
    if violated:
        return ()
    return (criterion_id,)


def _folded_result(
    parent: CriterionFinding,
    *,
    in_denominator: bool,
    satisfied: bool,
    unresolved: bool,
    violated: tuple[str, ...],
) -> FindingResult:
    if violated:
        return FindingResult.CONTRADICTED
    if not in_denominator and parent.applicability is Applicability.UNCERTAIN:
        return FindingResult.INSUFFICIENT
    if not in_denominator:
        return FindingResult.UNMET
    if unresolved:
        return FindingResult.INSUFFICIENT
    if satisfied:
        return FindingResult.MET
    return FindingResult.UNMET


def _assess(criterion: Criterion, evidence: Evidence) -> CriterionAssessment:
    """Fold a top-level criterion and its subrequirements into one credit unit."""

    parent = _require_finding(evidence, criterion.id)
    pairs = tuple(
        (subrequirement, _require_finding(evidence, subrequirement.id))
        for subrequirement in criterion.subrequirements
    )
    child_findings = tuple(finding for _, finding in pairs)
    findings = (parent, *child_findings)
    violated = _violated_ids(criterion.id, parent, pairs)
    child_in_force = any(
        finding.applicability is Applicability.APPLICABLE
        or finding.result is FindingResult.CONTRADICTED
        for finding in child_findings
    )
    in_denominator = (
        criterion.requirement_type is RequirementType.REQUIRED
        or parent.applicability is Applicability.APPLICABLE
        or child_in_force
    )
    satisfied = (
        in_denominator
        and not violated
        and _is_met(parent)
        and all(_child_clear(finding) for finding in child_findings)
    )
    established = _is_met(parent) or any(_is_met(finding) for finding in child_findings)
    unresolved = in_denominator and any(_is_unresolved_finding(finding) for finding in findings)
    if in_denominator or violated:
        applicability = Applicability.APPLICABLE
    else:
        applicability = parent.applicability
    return CriterionAssessment(
        criterion_id=criterion.id,
        applicability=applicability,
        result=_folded_result(
            parent,
            in_denominator=in_denominator,
            satisfied=satisfied,
            unresolved=unresolved,
            violated=violated,
        ),
        satisfied=satisfied,
        in_denominator=in_denominator,
        established=established,
        unresolved=unresolved,
        evidence_turns=_collect_turns(findings, _supports),
        open_turns=_collect_turns(findings, _is_open),
        missing_ids=_missing_ids(
            criterion.id,
            pairs,
            in_denominator=in_denominator,
            satisfied=satisfied,
            violated=violated,
        ),
        violated_ids=violated,
    )


def _require_coverage(workflow: Workflow, rubric: Rubric) -> None:
    missing = [node.id for node in workflow.nodes if not rubric.by_node.get(node.id)]
    if missing:
        fail(
            JudgeCode.INVALID_EVIDENCE,
            f"Every workflow node needs a criterion. Missing: {', '.join(missing)}.",
        )


def _predecessors(workflow: Workflow) -> Mapping[str, tuple[str, ...]]:
    buckets: dict[str, list[str]] = {node.id: [] for node in workflow.nodes}
    for edge in workflow.edges:
        sources = buckets[edge.target]
        if edge.source not in sources:
            sources.append(edge.source)
    return MappingProxyType({node_id: tuple(sources) for node_id, sources in buckets.items()})


def _descendants(outgoing: Mapping[str, tuple[str, ...]], node_id: str) -> frozenset[str]:
    seen: set[str] = set()
    pending: deque[str] = deque(outgoing[node_id])
    while pending:
        current = pending.popleft()
        if current in seen:
            continue
        seen.add(current)
        pending.extend(outgoing[current])
    return frozenset(seen)


def _criteria_complete(items: tuple[CriterionAssessment, ...]) -> bool:
    counted = 0
    for item in items:
        if item.violated_ids:
            return False
        if not item.in_denominator:
            continue
        counted += 1
        if not item.satisfied or item.unresolved:
            return False
    return counted > 0


def _turn_facts(transcript: Transcript) -> tuple[int | None, frozenset[int]]:
    last_user: int | None = None
    assistant: set[int] = set()
    for turn in transcript.turns:
        if turn.role == "user":
            last_user = turn.index
        elif turn.role == "assistant":
            assistant.add(turn.index)
    return last_user, frozenset(assistant)


def _scope(
    workflow: Workflow,
    grouped: Mapping[str, tuple[CriterionAssessment, ...]],
    transcript: Transcript,
) -> _Scope:
    last_user, assistant_turns = _turn_facts(transcript)
    descendants = {
        node.id: _descendants(workflow.outgoing, node.id) for node in workflow.nodes
    }
    return _Scope(
        predecessors=_predecessors(workflow),
        descendants=MappingProxyType(descendants),
        satisfied_nodes=frozenset(
            node_id
            for node_id, items in grouped.items()
            if any(item.satisfied for item in items)
        ),
        complete_nodes=frozenset(
            node_id for node_id, items in grouped.items() if _criteria_complete(items)
        ),
        end_ids=frozenset(node.id for node in workflow.nodes if node.type == "end"),
        last_user=last_user,
        assistant_turns=assistant_turns,
    )


def _in_force(criteria: tuple[Criterion, ...], items: tuple[CriterionAssessment, ...]) -> bool:
    for criterion in criteria:
        if criterion.requirement_type is RequirementType.REQUIRED:
            return True
    return any(
        item.applicability is Applicability.APPLICABLE or bool(item.violated_ids) for item in items
    )


def _conditional_node(criteria: tuple[Criterion, ...]) -> bool:
    return all(criterion.requirement_type is not RequirementType.REQUIRED for criterion in criteria)


def _later_non_end_work(node_id: str, scope: _Scope) -> bool:
    return any(
        descendant in scope.satisfied_nodes and descendant not in scope.end_ids
        for descendant in scope.descendants[node_id]
    )


def _end_closed(scope: _Scope) -> bool:
    return any(end_id in scope.satisfied_nodes for end_id in scope.end_ids)


def _cites_last_user(items: tuple[CriterionAssessment, ...], last_user: int | None) -> bool:
    if last_user is None:
        return False
    return any(last_user in item.open_turns for item in items)


def _assistant_after_trigger(
    items: tuple[CriterionAssessment, ...],
    assistant_turns: frozenset[int],
) -> bool:
    cited = [
        turn
        for item in items
        if item.in_denominator and not item.satisfied
        for turn in item.open_turns
    ]
    if not cited:
        return False
    trigger = max(cited)
    return any(turn > trigger for turn in assistant_turns)


def _opportunity(
    node_id: str,
    criteria: tuple[Criterion, ...],
    items: tuple[CriterionAssessment, ...],
    scope: _Scope,
) -> bool:
    if _later_non_end_work(node_id, scope):
        return True
    if _end_closed(scope) and _cites_last_user(items, scope.last_user):
        return True
    # A conditional trigger is actionable when an assistant spoke after it.
    return _conditional_node(criteria) and _assistant_after_trigger(items, scope.assistant_turns)


def _has_partial_progress(items: tuple[CriterionAssessment, ...]) -> bool:
    for item in items:
        if not item.in_denominator:
            continue
        # Cited insufficient evidence is an attempt. An unsettled child with no turns is not.
        if item.established or (item.unresolved and item.evidence_turns):
            return True
    return False


def _join(ids: tuple[str, ...]) -> str:
    return ", ".join(ids)


def _explanation(
    status: NodeStatus,
    applicability: Applicability,
    satisfied_count: int,
    applicable_count: int,
    missing: tuple[str, ...],
    violated: tuple[str, ...],
) -> str:
    if applicability is Applicability.UNCERTAIN:
        return "The transcript does not establish whether this step applied."
    if status is NodeStatus.NOT_REACHED:
        return "The conversation ended before this step."
    if status is NodeStatus.NOT_APPLICABLE:
        return "No triggering condition for this step occurred."
    if status is NodeStatus.SKIPPED:
        if missing:
            return f"Applicable requirements were omitted: {_join(missing)}."
        return "Applicable requirements were omitted."
    if status is NodeStatus.INCORRECT:
        sentence = f"A requirement was contradicted: {_join(violated)}."
        if applicable_count and satisfied_count:
            sentence = (
                f"{sentence} {satisfied_count} of {applicable_count} "
                "applicable requirements were met."
            )
        if missing:
            return f"{sentence} Missing: {_join(missing)}."
        return sentence
    if status is NodeStatus.COMPLETED:
        return f"All {applicable_count} applicable requirements were met."
    sentence = f"{satisfied_count} of {applicable_count} applicable requirements were met."
    if missing:
        return f"{sentence} Missing: {_join(missing)}."
    return sentence


def _attempted(status: NodeStatus) -> AttemptState:
    if (
        status is NodeStatus.COMPLETED
        or status is NodeStatus.PARTIAL
        or status is NodeStatus.INCORRECT
    ):
        return AttemptState.ATTEMPTED
    return AttemptState.NOT_ATTEMPTED


def _result(
    node_id: str,
    items: tuple[CriterionAssessment, ...],
    status: NodeStatus,
    applicability: Applicability,
) -> NodeResult:
    missing = _unique([item_id for item in items for item_id in item.missing_ids])
    violated = _unique([item_id for item in items for item_id in item.violated_ids])
    evidence: set[int] = set()
    satisfied_count = 0
    applicable_count = 0
    for item in items:
        evidence.update(item.evidence_turns)
        if not item.in_denominator:
            continue
        applicable_count += 1
        if item.satisfied:
            satisfied_count += 1
    return NodeResult(
        node_id=node_id,
        applicability=applicability,
        attempted=_attempted(status),
        criteria=items,
        status=status,
        evidence_turns=tuple(sorted(evidence)),
        missing_requirements=missing,
        violated_requirements=violated,
        satisfied_count=satisfied_count,
        applicable_count=applicable_count,
        explanation=_explanation(
            status,
            applicability,
            satisfied_count,
            applicable_count,
            missing,
            violated,
        ),
    )


def _classify_node(
    node_id: str,
    criteria: tuple[Criterion, ...],
    items: tuple[CriterionAssessment, ...],
    scope: _Scope,
) -> NodeResult:
    # Applicability first. A contradiction means the step was in force.
    if not _in_force(criteria, items):
        if any(item.applicability is Applicability.UNCERTAIN for item in items):
            return _result(node_id, items, NodeStatus.NOT_REACHED, Applicability.UNCERTAIN)
        if any(source in scope.complete_nodes for source in scope.predecessors[node_id]):
            return _result(node_id, items, NodeStatus.NOT_APPLICABLE, Applicability.NOT_APPLICABLE)
        return _result(node_id, items, NodeStatus.NOT_REACHED, Applicability.NOT_APPLICABLE)
    if any(item.violated_ids for item in items):
        return _result(node_id, items, NodeStatus.INCORRECT, Applicability.APPLICABLE)
    if _criteria_complete(items):
        return _result(node_id, items, NodeStatus.COMPLETED, Applicability.APPLICABLE)
    if _has_partial_progress(items):
        return _result(node_id, items, NodeStatus.PARTIAL, Applicability.APPLICABLE)
    if _opportunity(node_id, criteria, items, scope):
        return _result(node_id, items, NodeStatus.SKIPPED, Applicability.APPLICABLE)
    return _result(node_id, items, NodeStatus.NOT_REACHED, Applicability.APPLICABLE)


def _conditional_nodes(rubric: Rubric) -> Mapping[str, str]:
    return MappingProxyType(
        {
            criterion.id: criterion.node_id
            for criterion in rubric.criteria
            if criterion.requirement_type is RequirementType.CONDITIONAL
        }
    )


def _entered_before(
    kept: tuple[StageEvent, ...] | list[StageEvent],
    sources: tuple[str, ...],
    event: StageEvent,
) -> bool:
    source_ids = set(sources)
    for prior in kept:
        if prior.node_id not in source_ids:
            continue
        if (prior.turn, prior.start) < (event.turn, event.start):
            return True
    return False


def _reachable_events(workflow: Workflow, events: tuple[StageEvent, ...]) -> tuple[StageEvent, ...]:
    """Keep stage entries that occur after a predecessor stage has been entered."""

    predecessors = _predecessors(workflow)
    kept: list[StageEvent] = []
    for event in sorted(events, key=lambda item: (item.turn, item.start, item.node_id)):
        sources = predecessors.get(event.node_id)
        if sources is None:
            continue
        if not sources or _entered_before(kept, sources, event):
            kept.append(event)
    return tuple(kept)


def _reach_turn(
    predecessors: Mapping[str, tuple[str, ...]],
    events: tuple[StageEvent, ...],
    node_id: str,
) -> int | None:
    sources = predecessors.get(node_id, ())
    if not sources:
        return 0
    turns = [event.turn for event in events if event.node_id in sources]
    if not turns:
        return None
    return min(turns)


def _cited_before_reach(finding: CriterionFinding, reach: int | None) -> bool:
    if finding.applicability is not Applicability.APPLICABLE:
        return False
    if reach is None:
        return True
    if finding.action_turn is not None and finding.action_turn < reach:
        return True
    return bool(finding.evidence_turns) and max(finding.evidence_turns) < reach


def _withdraw_condition(finding: CriterionFinding) -> CriterionFinding:
    return CriterionFinding(
        criterion_id=finding.criterion_id,
        applicability=Applicability.NOT_APPLICABLE,
        result=FindingResult.UNMET,
        evidence_turns=(),
        action_turn=None,
        reason="This condition is cited before the workflow reaches this step.",
    )


def _ground_findings(workflow: Workflow, rubric: Rubric, evidence: Evidence) -> Evidence:
    """Withdraw a conditional finding cited before its node can be reached.

    A conditional step applies only after a predecessor stage has been entered.
    An earlier repair does not establish that condition.
    """

    conditional = _conditional_nodes(rubric)
    if not conditional:
        return evidence
    predecessors = _predecessors(workflow)
    events = _reachable_events(workflow, evidence.stage_events)
    findings: list[CriterionFinding] = []
    changed = False
    for finding in evidence.findings:
        node_id = conditional.get(finding.criterion_id)
        if node_id is None:
            findings.append(finding)
            continue
        if _cited_before_reach(finding, _reach_turn(predecessors, events, node_id)):
            findings.append(_withdraw_condition(finding))
            changed = True
            continue
        findings.append(finding)
    if not changed:
        return evidence
    return Evidence(
        findings=tuple(findings),
        by_id=MappingProxyType({item.criterion_id: item for item in findings}),
        stage_events=events,
    )


def classify_nodes(
    workflow: Workflow,
    rubric: Rubric,
    evidence: Evidence,
    transcript: Transcript,
) -> NodeEvaluation:
    """Return one deterministic result for every workflow node."""

    _require_coverage(workflow, rubric)
    evidence = _ground_findings(workflow, rubric, evidence)
    grouped = {
        node.id: tuple(_assess(criterion, evidence) for criterion in rubric.by_node[node.id])
        for node in workflow.nodes
    }
    scope = _scope(workflow, grouped, transcript)
    results = tuple(
        _classify_node(node.id, rubric.by_node[node.id], grouped[node.id], scope)
        for node in workflow.nodes
    )
    return NodeEvaluation(
        results=results,
        by_id=MappingProxyType({result.node_id: result for result in results}),
    )


def _complete_paths(workflow: Workflow) -> tuple[tuple[str, ...], ...]:
    """List every simple path from the start node to an end node, in edge order."""

    end_ids = frozenset(node.id for node in workflow.nodes if node.type == "end")
    outgoing = workflow.outgoing
    pending: list[tuple[str, tuple[str, ...]]] = [(workflow.start_id, (workflow.start_id,))]
    found: list[tuple[str, ...]] = []
    while pending:
        node_id, path = pending.pop()
        if node_id in end_ids:
            found.append(path)
            continue
        for target in reversed(outgoing[node_id]):
            if target in path:
                continue
            pending.append((target, (*path, target)))
    return tuple(found)


def _branch_nodes(paths: tuple[tuple[str, ...], ...]) -> tuple[str, ...]:
    """Nodes that some complete path contains and another omits, in first-seen order."""

    order: list[str] = []
    seen: set[str] = set()
    for path in paths:
        for node_id in path:
            if node_id in seen:
                continue
            seen.add(node_id)
            order.append(node_id)
    presence = {node_id: 0 for node_id in order}
    total = len(paths)
    for path in paths:
        for node_id in set(path):
            presence[node_id] += 1
    return tuple(node_id for node_id in order if 0 < presence[node_id] < total)


def _expected_path(
    paths: tuple[tuple[str, ...], ...],
    evaluation: NodeEvaluation,
) -> tuple[str, ...] | None:
    """Select the one complete path the established branch nodes allow."""

    branches = _branch_nodes(paths)
    uncertain = False
    viable: list[tuple[str, ...]] = []
    for path in paths:
        path_nodes = set(path)
        keep = True
        for node_id in branches:
            applicability = evaluation.by_id[node_id].applicability
            on_path = node_id in path_nodes
            if applicability is Applicability.UNCERTAIN:
                uncertain = True
                continue
            if applicability is Applicability.APPLICABLE and not on_path:
                keep = False
                break
            if applicability is Applicability.NOT_APPLICABLE and on_path:
                keep = False
                break
        if keep:
            viable.append(path)
    if uncertain or len(viable) != 1:
        return None
    return viable[0]


def _assess_transition(
    source: str,
    target: str,
    workflow: Workflow,
    expected: tuple[str, ...] | None,
) -> TransitionAssessment:
    if target not in workflow.outgoing[source]:
        return TransitionAssessment.INVALID
    if expected is None:
        return TransitionAssessment.UNRESOLVED
    for index, node_id in enumerate(expected):
        if node_id != source:
            continue
        following = expected[index + 1] if index + 1 < len(expected) else None
        if following == target:
            return TransitionAssessment.PERMITTED
        return TransitionAssessment.INVALID
    return TransitionAssessment.INVALID


def _ordered_events(events: tuple[StageEvent, ...]) -> tuple[tuple[StageEvent, ...], bool]:
    """Sort by turn and offset. A shared turn and offset is an unordered tie."""

    ordered = tuple(sorted(events, key=lambda event: (event.turn, event.start)))
    for index in range(1, len(ordered)):
        previous = ordered[index - 1]
        current = ordered[index]
        if (
            previous.turn == current.turn
            and previous.start == current.start
            and previous.node_id != current.node_id
        ):
            return ordered[: index - 1], True
    return ordered, False


def _fold_events(
    workflow: Workflow,
    events: tuple[StageEvent, ...],
    expected: tuple[str, ...] | None,
) -> _Fold:
    ordered, tied = _ordered_events(events)
    end_ids = frozenset(node.id for node in workflow.nodes if node.type == "end")
    descendants = {
        node.id: _descendants(workflow.outgoing, node.id) for node in workflow.nodes
    }
    path: list[str] = []
    transitions: list[ObservedTransition] = []
    current: str | None = None
    end_seen = False
    for event in ordered:
        node_id = event.node_id
        if current is None:
            path.append(node_id)
            current = node_id
            end_seen = node_id in end_ids
            continue
        # Another action in the current stage is not a new visit.
        if node_id == current:
            continue
        # A return to an open ancestor, before any end node, continues that stage.
        if node_id in path and not end_seen and current in descendants[node_id]:
            continue
        transitions.append(
            ObservedTransition(
                source=current,
                target=node_id,
                assessment=_assess_transition(current, node_id, workflow, expected),
            )
        )
        path.append(node_id)
        current = node_id
        if node_id in end_ids:
            end_seen = True
    return _Fold(observed=tuple(path), transitions=tuple(transitions), tied=tied)


def _is_prefix(observed: tuple[str, ...], expected: tuple[str, ...]) -> bool:
    return expected[: len(observed)] == observed


def _on_end(observed: tuple[str, ...], end_ids: frozenset[str]) -> bool:
    return bool(observed) and observed[-1] in end_ids


def _validity(
    observed: tuple[str, ...],
    expected: tuple[str, ...] | None,
    transitions: tuple[ObservedTransition, ...],
    tied: bool,
    end_ids: frozenset[str],
) -> PathValidity:
    if any(item.assessment is TransitionAssessment.INVALID for item in transitions):
        return PathValidity.INVALID
    if tied or expected is None:
        return PathValidity.INDETERMINATE
    if observed == expected:
        return PathValidity.VALID
    if _is_prefix(observed, expected) and not _on_end(observed, end_ids):
        return PathValidity.INCOMPLETE
    return PathValidity.INVALID


def _gap_nodes(
    expected: tuple[str, ...] | None,
    observed: tuple[str, ...],
    evaluation: NodeEvaluation,
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    if expected is None:
        return (), ()
    observed_nodes = set(observed)
    later = [False] * len(expected)
    seen_later = False
    for index in range(len(expected) - 1, -1, -1):
        later[index] = seen_later
        if expected[index] in observed_nodes:
            seen_later = True
    omitted: list[str] = []
    unreached: list[str] = []
    for index, node_id in enumerate(expected):
        if node_id in observed_nodes:
            continue
        status = evaluation.by_id[node_id].status
        if status is NodeStatus.NOT_REACHED:
            unreached.append(node_id)
            continue
        if status is NodeStatus.SKIPPED or later[index]:
            omitted.append(node_id)
    return tuple(omitted), tuple(unreached)


def verify_traversal(
    workflow: Workflow,
    evaluation: NodeEvaluation,
    evidence: Evidence,
) -> Traversal:
    """Reconstruct the stage progression and check it against the graph."""

    paths = _complete_paths(workflow)
    expected = _expected_path(paths, evaluation)
    folded = _fold_events(workflow, _reachable_events(workflow, evidence.stage_events), expected)
    end_ids = frozenset(node.id for node in workflow.nodes if node.type == "end")
    omitted, unreached = _gap_nodes(expected, folded.observed, evaluation)
    validity = _validity(
        folded.observed,
        expected,
        folded.transitions,
        folded.tied,
        end_ids,
    )
    return Traversal(
        candidate_paths=paths,
        expected_path=expected,
        observed_path=folded.observed,
        transitions=folded.transitions,
        omitted_nodes=omitted,
        unreached_nodes=unreached,
        validity=validity,
        complete=validity is PathValidity.VALID,
        order_tied=folded.tied,
    )


_POINTS = Decimal(100)
_CENT = Decimal("0.01")


def _ratio(numerator: int, denominator: int) -> Decimal:
    return Decimal(numerator) / Decimal(denominator)


def _node_ratio(result: NodeResult) -> Decimal | None:
    """Credit for one node. Inapplicable and unreached nodes have no ratio."""

    if result.status is NodeStatus.NOT_APPLICABLE or result.status is NodeStatus.NOT_REACHED:
        return None
    if result.applicable_count == 0:
        return None
    return _ratio(result.satisfied_count, result.applicable_count)


def _node_scores(evaluation: NodeEvaluation) -> tuple[NodeScore, ...]:
    return tuple(
        NodeScore(node_id=result.node_id, score=_node_ratio(result))
        for result in evaluation.results
    )


def _established_path(
    evaluation: NodeEvaluation,
    traversal: Traversal,
) -> tuple[str, ...] | None:
    """The selected branch, when applicability and traversal are both settled."""

    expected = traversal.expected_path
    if expected is None or traversal.validity is PathValidity.INDETERMINATE:
        return None
    for node_id in expected:
        if evaluation.by_id[node_id].applicability is Applicability.UNCERTAIN:
            return None
    return expected


def _expected_ratios(
    evaluation: NodeEvaluation,
    traversal: Traversal,
) -> tuple[Decimal, ...] | None:
    """Ratios for the established branch. Unresolved branches are not averaged."""

    expected = _established_path(evaluation, traversal)
    if expected is None:
        return None
    ratios: list[Decimal] = []
    for node_id in expected:
        ratio = _node_ratio(evaluation.by_id[node_id])
        if ratio is None:
            continue
        ratios.append(ratio)
    if not ratios:
        return None
    return tuple(ratios)


def _mean(ratios: tuple[Decimal, ...]) -> Decimal:
    return sum(ratios, start=Decimal(0)) / Decimal(len(ratios))


def _transition_score(traversal: Traversal) -> Decimal:
    observed = len(traversal.transitions)
    if observed == 0:
        # No observed transition leaves this factor at 1.
        # The verdict still requires a complete permitted path.
        return Decimal(1)
    permitted = 0
    for transition in traversal.transitions:
        if transition.assessment is TransitionAssessment.PERMITTED:
            permitted += 1
    return _ratio(permitted, observed)


def _points(node_average: Decimal, transition_score: Decimal) -> Decimal:
    scaled = node_average * transition_score * _POINTS
    return scaled.quantize(_CENT, rounding=ROUND_HALF_UP)


def _material_failure(evaluation: NodeEvaluation, traversal: Traversal) -> bool:
    if traversal.validity is PathValidity.INVALID:
        return True
    return any(result.status is NodeStatus.INCORRECT for result in evaluation.results)


def _unresolved(evaluation: NodeEvaluation, traversal: Traversal) -> bool:
    return _established_path(evaluation, traversal) is None


def _missing_credit(evaluation: NodeEvaluation, traversal: Traversal) -> bool:
    expected = traversal.expected_path
    if expected is None:
        return False
    for node_id in expected:
        result = evaluation.by_id[node_id]
        if result.status is NodeStatus.PARTIAL or result.status is NodeStatus.SKIPPED:
            return True
        ratio = _node_ratio(result)
        if ratio is not None and ratio < Decimal(1):
            return True
    return False


def _has_unreached_expected(evaluation: NodeEvaluation, traversal: Traversal) -> bool:
    expected = traversal.expected_path
    if expected is None:
        return False
    return any(
        evaluation.by_id[node_id].status is NodeStatus.NOT_REACHED for node_id in expected
    )


def _verdict(evaluation: NodeEvaluation, traversal: Traversal) -> Verdict:
    """Prefer a material failure, then an unresolved branch, then an unfinished path."""

    if _material_failure(evaluation, traversal):
        return Verdict.FAILED
    if _unresolved(evaluation, traversal):
        return Verdict.INDETERMINATE
    if traversal.validity is not PathValidity.VALID:
        return Verdict.INCOMPLETE
    if _missing_credit(evaluation, traversal):
        return Verdict.PARTIAL
    # An expected stage that was never reached is not a completed outcome.
    if _has_unreached_expected(evaluation, traversal):
        return Verdict.INCOMPLETE
    return Verdict.COMPLETED


def score_adherence(evaluation: NodeEvaluation, traversal: Traversal) -> Adherence:
    """Score criterion completion and permitted progression, then assign a verdict."""

    ratios = _expected_ratios(evaluation, traversal)
    transition_score = _transition_score(traversal)
    if ratios is None:
        node_average = None
        adherence_score = None
    else:
        node_average = _mean(ratios)
        adherence_score = _points(node_average, transition_score)
    return Adherence(
        nodes=_node_scores(evaluation),
        node_average=node_average,
        transition_score=transition_score,
        adherence_score=adherence_score,
        verdict=_verdict(evaluation, traversal),
    )


__all__ = [
    "Adherence",
    "AttemptState",
    "CriterionAssessment",
    "NodeEvaluation",
    "NodeResult",
    "NodeScore",
    "NodeStatus",
    "ObservedTransition",
    "PathValidity",
    "TransitionAssessment",
    "Traversal",
    "Verdict",
    "classify_nodes",
    "score_adherence",
    "verify_traversal",
]
