"""Check the observed stage progression against the directed graph."""

from __future__ import annotations

from dataclasses import dataclass

from .inputs import Workflow
from .models import (
    Applicability,
    Evidence,
    NodeEvaluation,
    NodeStatus,
    ObservedTransition,
    PathValidity,
    StageEvent,
    TransitionAssessment,
    Traversal,
)
from .nodes import descendants, reachable_events


@dataclass(frozen=True, slots=True)
class _Fold:
    observed: tuple[str, ...]
    transitions: tuple[ObservedTransition, ...]
    tied: bool


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
    descendants_by_node = {
        node.id: descendants(workflow.outgoing, node.id) for node in workflow.nodes
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
        if node_id in path and not end_seen and current in descendants_by_node[node_id]:
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
    folded = _fold_events(workflow, reachable_events(workflow, evidence.stage_events), expected)
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


__all__ = ["verify_traversal"]
