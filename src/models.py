"""Records shared by more than one pipeline stage."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from decimal import Decimal
from enum import StrEnum
from typing import Protocol

from .utils import JsonValue


class RequirementType(StrEnum):
    REQUIRED = "required"
    CONDITIONAL = "conditional"
    ALTERNATIVE = "alternative"


class Applicability(StrEnum):
    APPLICABLE = "applicable"
    NOT_APPLICABLE = "not_applicable"
    UNCERTAIN = "uncertain"


class FindingResult(StrEnum):
    MET = "met"
    UNMET = "unmet"
    CONTRADICTED = "contradicted"
    INSUFFICIENT = "insufficient"


@dataclass(frozen=True, slots=True)
class Subrequirement:
    id: str
    description: str
    requirement_type: RequirementType
    condition: str | None
    alternative_group: str | None
    exact_wording: bool
    expected_behavior: str
    source_text: str


@dataclass(frozen=True, slots=True)
class Criterion:
    id: str
    node_id: str
    description: str
    requirement_type: RequirementType
    condition: str | None
    alternative_group: str | None
    exact_wording: bool
    expected_behavior: str
    source_text: str
    subrequirements: tuple[Subrequirement, ...]


@dataclass(frozen=True, slots=True)
class Rubric:
    criteria: tuple[Criterion, ...]
    by_node: Mapping[str, tuple[Criterion, ...]]


@dataclass(frozen=True, slots=True)
class CriterionFinding:
    criterion_id: str
    applicability: Applicability
    result: FindingResult
    evidence_turns: tuple[int, ...]
    action_turn: int | None
    reason: str


@dataclass(frozen=True, slots=True)
class StageEvent:
    """An assistant action that enters a workflow stage."""

    node_id: str
    turn: int
    start: int
    anchor: str


@dataclass(frozen=True, slots=True)
class Evidence:
    findings: tuple[CriterionFinding, ...]
    by_id: Mapping[str, CriterionFinding]
    stage_events: tuple[StageEvent, ...]


class StructuredCompleter(Protocol):
    """One structured model call that returns JSON text."""

    def complete(
        self,
        *,
        instructions: str,
        content: str,
        schema_name: str,
        schema: Mapping[str, JsonValue],
        label: str,
    ) -> str:
        """Return the model's structured JSON document."""
        ...


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


__all__ = [
    "Adherence",
    "Applicability",
    "AttemptState",
    "Criterion",
    "CriterionAssessment",
    "CriterionFinding",
    "Evidence",
    "FindingResult",
    "NodeEvaluation",
    "NodeResult",
    "NodeScore",
    "NodeStatus",
    "ObservedTransition",
    "PathValidity",
    "RequirementType",
    "Rubric",
    "StageEvent",
    "StructuredCompleter",
    "Subrequirement",
    "TransitionAssessment",
    "Traversal",
    "Verdict",
]
