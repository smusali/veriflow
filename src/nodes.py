"""Classify each workflow node from rubric findings."""

from __future__ import annotations

from collections import deque
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from types import MappingProxyType

from .errors import JudgeCode, fail_judge as fail
from .inputs import Transcript, Workflow, build_incoming
from .models import (
    Applicability,
    AttemptState,
    Criterion,
    CriterionAssessment,
    CriterionFinding,
    Evidence,
    FindingResult,
    NodeEvaluation,
    NodeResult,
    NodeStatus,
    RequirementType,
    Rubric,
    StageEvent,
    Subrequirement,
)


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
    return finding.applicability is Applicability.APPLICABLE and finding.result is FindingResult.MET


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
    applicability = Applicability.APPLICABLE if in_denominator or violated else parent.applicability
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


def descendants(outgoing: Mapping[str, tuple[str, ...]], node_id: str) -> frozenset[str]:
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
    descendants_by_node = {
        node.id: descendants(workflow.outgoing, node.id) for node in workflow.nodes
    }
    return _Scope(
        predecessors=build_incoming(workflow),
        descendants=MappingProxyType(descendants_by_node),
        satisfied_nodes=frozenset(
            node_id for node_id, items in grouped.items() if any(item.satisfied for item in items)
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


def reachable_events(workflow: Workflow, events: tuple[StageEvent, ...]) -> tuple[StageEvent, ...]:
    """Keep stage entries that occur after a predecessor stage has been entered."""

    predecessors = build_incoming(workflow)
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
    predecessors = build_incoming(workflow)
    events = reachable_events(workflow, evidence.stage_events)
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


__all__ = ["classify_nodes", "descendants", "reachable_events"]
