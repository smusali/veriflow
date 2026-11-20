"""Assemble a machine-readable evaluation from node, path, and score results."""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from decimal import Decimal
from enum import StrEnum
from types import MappingProxyType

from .inputs import Workflow
from .judge import Applicability, CriterionFinding, Evidence, FindingResult
from .verifier import (
    Adherence,
    CriterionAssessment,
    NodeEvaluation,
    NodeResult,
    NodeStatus,
    PathValidity,
    TransitionAssessment,
    Traversal,
    Verdict,
)


class DeviationKind(StrEnum):
    MISSING_REQUIREMENT = "missing_requirement"
    VIOLATED_REQUIREMENT = "violated_requirement"
    INVALID_TRANSITION = "invalid_transition"


class UncertaintyKind(StrEnum):
    BRANCH = "branch"
    ORDER = "order"
    APPLICABILITY = "applicability"
    EVIDENCE = "evidence"
    TRANSITION = "transition"


@dataclass(frozen=True, slots=True)
class OverallReport:
    adherence_score: Decimal | None
    verdict: Verdict
    path_validity: PathValidity


@dataclass(frozen=True, slots=True)
class CriterionReport:
    criterion_id: str
    applicability: Applicability
    result: FindingResult
    evidence_turns: tuple[int, ...]
    missing_requirements: tuple[str, ...]
    violated_requirements: tuple[str, ...]
    reason: str


@dataclass(frozen=True, slots=True)
class NodeReport:
    status: NodeStatus
    applicability: Applicability
    score: Decimal | None
    evidence_turns: tuple[int, ...]
    missing_requirements: tuple[str, ...]
    violated_requirements: tuple[str, ...]
    explanation: str
    criteria: tuple[CriterionReport, ...]


@dataclass(frozen=True, slots=True)
class TransitionReport:
    source: str
    target: str
    assessment: TransitionAssessment


@dataclass(frozen=True, slots=True)
class Deviation:
    kind: DeviationKind
    node_id: str
    criterion: str | None
    source: str | None
    target: str | None
    reason: str


@dataclass(frozen=True, slots=True)
class Uncertainty:
    kind: UncertaintyKind
    node_id: str | None
    criterion: str | None
    source: str | None
    target: str | None
    reason: str


@dataclass(frozen=True, slots=True)
class EvaluationReport:
    overall: OverallReport
    expected_path: tuple[str, ...] | None
    observed_path: tuple[str, ...]
    transitions: tuple[TransitionReport, ...]
    node_results: Mapping[str, NodeReport]
    deviations: tuple[Deviation, ...]
    uncertainty: tuple[Uncertainty, ...]


def _report_node_ids(evaluation: NodeEvaluation) -> tuple[str, ...]:
    return tuple(result.node_id for result in evaluation.results)


def _owner(criterion_id: str, node_ids: tuple[str, ...]) -> str:
    # A criterion id starts with its node id and a dot. The longest match wins.
    match: str | None = None
    for node_id in node_ids:
        prefix = f"{node_id}."
        if criterion_id.startswith(prefix) and (match is None or len(node_id) > len(match)):
            match = node_id
    if match is None:
        raise ValueError(f"{criterion_id} does not belong to a workflow node.")
    return match


def _finding_reason(evidence: Evidence, criterion_id: str) -> str:
    finding = evidence.by_id.get(criterion_id)
    if finding is None:
        raise ValueError(f"{criterion_id} has no finding.")
    return finding.reason


def _scores(adherence: Adherence) -> Mapping[str, Decimal | None]:
    return MappingProxyType({node.node_id: node.score for node in adherence.nodes})


def _node_score(scores: Mapping[str, Decimal | None], node_id: str) -> Decimal | None:
    if node_id not in scores:
        raise ValueError(f"Node {node_id} has no score.")
    return scores[node_id]


def _reported(item: CriterionAssessment) -> bool:
    return bool(
        item.in_denominator
        or item.unresolved
        or item.missing_ids
        or item.violated_ids
        or item.evidence_turns
    )


def _criteria(result: NodeResult, evidence: Evidence) -> tuple[CriterionReport, ...]:
    selected: list[CriterionReport] = []
    for item in result.criteria:
        if not _reported(item):
            continue
        selected.append(
            CriterionReport(
                criterion_id=item.criterion_id,
                applicability=item.applicability,
                result=item.result,
                evidence_turns=item.evidence_turns,
                missing_requirements=item.missing_ids,
                violated_requirements=item.violated_ids,
                reason=_finding_reason(evidence, item.criterion_id),
            )
        )
    return tuple(selected)


def _node_report(
    result: NodeResult,
    scores: Mapping[str, Decimal | None],
    evidence: Evidence,
) -> NodeReport:
    return NodeReport(
        status=result.status,
        applicability=result.applicability,
        score=_node_score(scores, result.node_id),
        evidence_turns=result.evidence_turns,
        missing_requirements=result.missing_requirements,
        violated_requirements=result.violated_requirements,
        explanation=result.explanation,
        criteria=_criteria(result, evidence),
    )


def _is_missing(finding: CriterionFinding) -> bool:
    return (
        finding.applicability is Applicability.APPLICABLE
        and finding.result is FindingResult.UNMET
    )


def _is_violated(finding: CriterionFinding) -> bool:
    return finding.result is FindingResult.CONTRADICTED


def _is_uncertain(finding: CriterionFinding) -> bool:
    if finding.result is FindingResult.CONTRADICTED:
        return False
    if finding.applicability is Applicability.UNCERTAIN:
        return True
    return finding.result is FindingResult.INSUFFICIENT


def _matching(
    findings: tuple[CriterionFinding, ...],
    node_ids: tuple[str, ...],
    include: Callable[[CriterionFinding], bool],
) -> tuple[tuple[str, CriterionFinding], ...]:
    grouped: dict[str, list[CriterionFinding]] = {node_id: [] for node_id in node_ids}
    for finding in findings:
        if not include(finding):
            continue
        owner = _owner(finding.criterion_id, node_ids)
        grouped[owner].append(finding)
    ordered: list[tuple[str, CriterionFinding]] = []
    for node_id in node_ids:
        for finding in grouped[node_id]:
            ordered.append((node_id, finding))
    return tuple(ordered)


def _requirement_note(kind: DeviationKind, node_id: str, finding: CriterionFinding) -> Deviation:
    return Deviation(
        kind=kind,
        node_id=node_id,
        criterion=finding.criterion_id,
        source=None,
        target=None,
        reason=finding.reason,
    )


def _established_successor(expected: tuple[str, ...] | None, source: str) -> str | None:
    if expected is None:
        return None
    for index, node_id in enumerate(expected):
        if node_id != source:
            continue
        if index + 1 == len(expected):
            return None
        return expected[index + 1]
    return None


def _deviation_kind(finding: CriterionFinding) -> DeviationKind | None:
    if _is_violated(finding):
        return DeviationKind.VIOLATED_REQUIREMENT
    if _is_missing(finding):
        return DeviationKind.MISSING_REQUIREMENT
    return None


def _is_deviation(finding: CriterionFinding) -> bool:
    return _deviation_kind(finding) is not None


def _invalid_reason(
    source: str,
    target: str,
    workflow: Workflow,
    expected: tuple[str, ...] | None,
) -> str:
    """Explain an invalid hop from the graph edge and the established successor."""

    if target not in workflow.outgoing.get(source, ()):
        return f"The graph has no edge from {source} to {target}."
    if _established_successor(expected, source) != target:
        return f"The transition from {source} to {target} leaves the established branch."
    raise ValueError(f"The transition from {source} to {target} is an established branch step.")


def _deviations(
    workflow: Workflow,
    evaluation: NodeEvaluation,
    traversal: Traversal,
    evidence: Evidence,
) -> tuple[Deviation, ...]:
    node_ids = _report_node_ids(evaluation)
    notes: list[Deviation] = []
    for node_id, finding in _matching(evidence.findings, node_ids, _is_deviation):
        kind = _deviation_kind(finding)
        if kind is None:
            continue
        notes.append(_requirement_note(kind, node_id, finding))
    for hop in traversal.transitions:
        if hop.assessment is not TransitionAssessment.INVALID:
            continue
        notes.append(
            Deviation(
                kind=DeviationKind.INVALID_TRANSITION,
                node_id=hop.source,
                criterion=None,
                source=hop.source,
                target=hop.target,
                reason=_invalid_reason(hop.source, hop.target, workflow, traversal.expected_path),
            )
        )
    return tuple(notes)


def _evidence_note(node_id: str, finding: CriterionFinding) -> Uncertainty:
    return Uncertainty(
        kind=UncertaintyKind.EVIDENCE,
        node_id=node_id,
        criterion=finding.criterion_id,
        source=None,
        target=None,
        reason=finding.reason,
    )


def _uncertainty(
    evaluation: NodeEvaluation,
    traversal: Traversal,
    evidence: Evidence,
) -> tuple[Uncertainty, ...]:
    notes: list[Uncertainty] = []
    if traversal.expected_path is None:
        notes.append(
            Uncertainty(
                kind=UncertaintyKind.BRANCH,
                node_id=None,
                criterion=None,
                source=None,
                target=None,
                reason="The transcript does not establish which branch applies.",
            )
        )
    if traversal.order_tied:
        notes.append(
            Uncertainty(
                kind=UncertaintyKind.ORDER,
                node_id=None,
                criterion=None,
                source=None,
                target=None,
                reason="Two stage actions in one turn have no observable order.",
            )
        )
    for result in evaluation.results:
        if result.applicability is not Applicability.UNCERTAIN:
            continue
        notes.append(
            Uncertainty(
                kind=UncertaintyKind.APPLICABILITY,
                node_id=result.node_id,
                criterion=None,
                source=None,
                target=None,
                reason=result.explanation,
            )
        )
    for node_id, finding in _matching(
        evidence.findings,
        _report_node_ids(evaluation),
        _is_uncertain,
    ):
        notes.append(_evidence_note(node_id, finding))
    for hop in traversal.transitions:
        if hop.assessment is not TransitionAssessment.UNRESOLVED:
            continue
        notes.append(
            Uncertainty(
                kind=UncertaintyKind.TRANSITION,
                node_id=hop.source,
                criterion=None,
                source=hop.source,
                target=hop.target,
                reason=(
                    f"The transition from {hop.source} to {hop.target} "
                    "cannot be checked until the branch is established."
                ),
            )
        )
    return tuple(notes)


def assemble_report(
    workflow: Workflow,
    evaluation: NodeEvaluation,
    traversal: Traversal,
    adherence: Adherence,
    evidence: Evidence,
) -> EvaluationReport:
    """Combine node findings, traversal, and the adherence score into one report."""

    scores = _scores(adherence)
    node_results = {
        result.node_id: _node_report(result, scores, evidence) for result in evaluation.results
    }
    return EvaluationReport(
        overall=OverallReport(
            adherence_score=adherence.adherence_score,
            verdict=adherence.verdict,
            path_validity=traversal.validity,
        ),
        expected_path=traversal.expected_path,
        observed_path=traversal.observed_path,
        transitions=tuple(
            TransitionReport(source=hop.source, target=hop.target, assessment=hop.assessment)
            for hop in traversal.transitions
        ),
        node_results=MappingProxyType(node_results),
        deviations=_deviations(workflow, evaluation, traversal, evidence),
        uncertainty=_uncertainty(evaluation, traversal, evidence),
    )


def _text(value: str) -> str:
    return json.dumps(value, ensure_ascii=False)


def _text_or_null(value: str | None) -> str:
    if value is None:
        return "null"
    return _text(value)


def _enum(member: StrEnum) -> str:
    return _text(member.value)


def _integer(value: int) -> str:
    if isinstance(value, bool):
        raise ValueError("A turn index must be an integer.")
    return str(value)


def _decimal(value: Decimal) -> str:
    if not value.is_finite():
        raise ValueError("A score must be a finite number.")
    return format(value, "f")


def _decimal_or_null(value: Decimal | None) -> str:
    if value is None:
        return "null"
    return _decimal(value)


def _object(pairs: tuple[tuple[str, str], ...], level: int) -> str:
    if not pairs:
        return "{}"
    pad = "  " * level
    inner = "  " * (level + 1)
    lines = [f"{inner}{_text(key)}: {token}" for key, token in pairs]
    return "{\n" + ",\n".join(lines) + "\n" + pad + "}"


def _array(tokens: tuple[str, ...], level: int) -> str:
    if not tokens:
        return "[]"
    pad = "  " * level
    inner = "  " * (level + 1)
    lines = [f"{inner}{token}" for token in tokens]
    return "[\n" + ",\n".join(lines) + "\n" + pad + "]"


def _texts(values: tuple[str, ...], level: int) -> str:
    return _array(tuple(_text(value) for value in values), level)


def _path(values: tuple[str, ...] | None, level: int) -> str:
    if values is None:
        return "null"
    return _texts(values, level)


def _turns(values: tuple[int, ...], level: int) -> str:
    return _array(tuple(_integer(value) for value in values), level)


def _note(
    *,
    kind: StrEnum,
    node_id: str | None,
    criterion: str | None,
    source: str | None,
    target: str | None,
    reason: str,
    level: int,
) -> str:
    return _object(
        (
            ("kind", _enum(kind)),
            ("node_id", _text_or_null(node_id)),
            ("criterion", _text_or_null(criterion)),
            ("source", _text_or_null(source)),
            ("target", _text_or_null(target)),
            ("reason", _text(reason)),
        ),
        level,
    )


def _overall(report: OverallReport, level: int) -> str:
    return _object(
        (
            ("adherence_score", _decimal_or_null(report.adherence_score)),
            ("verdict", _enum(report.verdict)),
            ("path_validity", _enum(report.path_validity)),
        ),
        level,
    )


def _criterion(item: CriterionReport, level: int) -> str:
    return _object(
        (
            ("id", _text(item.criterion_id)),
            ("applicability", _enum(item.applicability)),
            ("result", _enum(item.result)),
            ("evidence_turns", _turns(item.evidence_turns, level + 1)),
            ("missing_requirements", _texts(item.missing_requirements, level + 1)),
            ("violated_requirements", _texts(item.violated_requirements, level + 1)),
            ("reason", _text(item.reason)),
        ),
        level,
    )


def _node(node: NodeReport, level: int) -> str:
    return _object(
        (
            ("status", _enum(node.status)),
            ("applicability", _enum(node.applicability)),
            ("score", _decimal_or_null(node.score)),
            ("evidence_turns", _turns(node.evidence_turns, level + 1)),
            ("missing_requirements", _texts(node.missing_requirements, level + 1)),
            ("violated_requirements", _texts(node.violated_requirements, level + 1)),
            ("explanation", _text(node.explanation)),
            (
                "criteria",
                _array(tuple(_criterion(item, level + 2) for item in node.criteria), level + 1),
            ),
        ),
        level,
    )


def _transition(item: TransitionReport, level: int) -> str:
    return _object(
        (
            ("source", _text(item.source)),
            ("target", _text(item.target)),
            ("assessment", _enum(item.assessment)),
        ),
        level,
    )


def _deviation(item: Deviation, level: int) -> str:
    return _note(
        kind=item.kind,
        node_id=item.node_id,
        criterion=item.criterion,
        source=item.source,
        target=item.target,
        reason=item.reason,
        level=level,
    )


def _uncertainty_note(item: Uncertainty, level: int) -> str:
    return _note(
        kind=item.kind,
        node_id=item.node_id,
        criterion=item.criterion,
        source=item.source,
        target=item.target,
        reason=item.reason,
        level=level,
    )


def render_report(report: EvaluationReport) -> str:
    """Serialize a report as JSON text."""

    return _object(
        (
            ("overall", _overall(report.overall, 1)),
            ("expected_path", _path(report.expected_path, 1)),
            ("observed_path", _texts(report.observed_path, 1)),
            (
                "transitions",
                _array(tuple(_transition(item, 2) for item in report.transitions), 1),
            ),
            (
                "node_results",
                _object(
                    tuple(
                        (node_id, _node(node, 2))
                        for node_id, node in report.node_results.items()
                    ),
                    1,
                ),
            ),
            (
                "deviations",
                _array(tuple(_deviation(item, 2) for item in report.deviations), 1),
            ),
            (
                "uncertainty",
                _array(tuple(_uncertainty_note(item, 2) for item in report.uncertainty), 1),
            ),
        ),
        0,
    )


__all__ = [
    "CriterionReport",
    "Deviation",
    "DeviationKind",
    "EvaluationReport",
    "NodeReport",
    "OverallReport",
    "TransitionReport",
    "Uncertainty",
    "UncertaintyKind",
    "assemble_report",
    "render_report",
]
