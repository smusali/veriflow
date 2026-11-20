"""Score criterion completion and permitted progression, and assign a verdict."""

from __future__ import annotations

from decimal import ROUND_HALF_UP, Decimal

from .models import (
    Adherence,
    Applicability,
    NodeEvaluation,
    NodeResult,
    NodeScore,
    NodeStatus,
    PathValidity,
    TransitionAssessment,
    Traversal,
    Verdict,
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

__all__ = ["score_adherence"]
