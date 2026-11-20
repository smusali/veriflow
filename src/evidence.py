"""Judge rubric obligations against a transcript."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import NoReturn

from .errors import JudgeCode, JudgeError, fail_judge as fail
from .inputs import Transcript
from .models import (
    Applicability,
    Criterion,
    CriterionFinding,
    Evidence,
    FindingResult,
    RequirementType,
    Rubric,
    StageEvent,
    StructuredCompleter,
    Subrequirement,
    complete_or_repair,
)
from .prompts import (
    EVIDENCE_INSTRUCTIONS as _EVIDENCE_INSTRUCTIONS,
    EVIDENCE_REPAIR_INSTRUCTIONS as _EVIDENCE_REPAIR_INSTRUCTIONS,
    FINDINGS_SCHEMA as _FINDINGS_SCHEMA,
)
from .utils import JsonReader, JsonValue, decode_document, enum_member

_FINDING_FIELDS = frozenset(
    {
        "criterion_id",
        "evidence_turns",
        "action_turn",
        "applicability",
        "result",
        "reason",
    }
)
_EVIDENCE_FIELDS = frozenset({"findings", "stage_events"})
_STAGE_EVENT_FIELDS = frozenset({"node_id", "turn", "start", "anchor"})


def _reject(message: str) -> NoReturn:
    fail(JudgeCode.INVALID_EVIDENCE, message)


_reader = JsonReader(_reject, reject_whitespace=True)


@dataclass(frozen=True, slots=True)
class _JudgedObligation:
    id: str
    node_id: str
    parent_id: str | None
    requirement_type: RequirementType
    condition: str | None
    alternative_group: str | None
    exact_wording: bool
    description: str
    expected_behavior: str


def _judged(
    item: Criterion | Subrequirement,
    *,
    node_id: str,
    parent_id: str | None,
) -> _JudgedObligation:
    return _JudgedObligation(
        id=item.id,
        node_id=node_id,
        parent_id=parent_id,
        requirement_type=item.requirement_type,
        condition=item.condition,
        alternative_group=item.alternative_group,
        exact_wording=item.exact_wording,
        description=item.description,
        expected_behavior=item.expected_behavior,
    )


def _judged_obligations(rubric: Rubric) -> tuple[_JudgedObligation, ...]:
    items: list[_JudgedObligation] = []
    for criterion in rubric.criteria:
        items.append(_judged(criterion, node_id=criterion.node_id, parent_id=None))
        for subrequirement in criterion.subrequirements:
            items.append(_judged(subrequirement, node_id=criterion.node_id, parent_id=criterion.id))
    return tuple(items)


def _group_scope(item: _JudgedObligation) -> str:
    if item.parent_id is None:
        return f"node {item.node_id}"
    return f"criterion {item.parent_id}"


def _require_offset(value: JsonValue, location: str) -> int:
    number = _reader.require_int(value, location, label="offset")
    if number < 0:
        fail(JudgeCode.INVALID_EVIDENCE, f"{location} {number} is negative.")
    return number


def _read_stage_event(
    value: JsonValue,
    index: int,
    transcript: Transcript,
    node_ids: frozenset[str],
) -> StageEvent:
    location = f"stage_events[{index}]"
    code = JudgeCode.INVALID_EVIDENCE
    fields = _reader.expect_exact_object(value, location, _STAGE_EVENT_FIELDS)
    node_id = _reader.require_text_field(fields, "node_id", location)
    if node_id not in node_ids:
        fail(code, f"{location}.node_id {node_id!r} is not a workflow node.")
    turn = _require_turn_index(
        _reader.require_field(fields, "turn", location),
        f"{location}.turn",
        len(transcript.turns),
    )
    spoken = transcript.turns[turn]
    if spoken.role != "assistant":
        fail(code, f"{location}.turn {turn} is not an assistant turn.")
    start = _require_offset(_reader.require_field(fields, "start", location), f"{location}.start")
    anchor = _reader.require_text_field(fields, "anchor", location)
    excerpt = spoken.content[start : start + len(anchor)]
    if excerpt != anchor:
        fail(code, f"{location}.anchor is not in the assistant turn at start {start}.")
    return StageEvent(node_id=node_id, turn=turn, start=start, anchor=anchor)


def _read_stage_events(
    value: JsonValue,
    transcript: Transcript,
    node_ids: frozenset[str],
) -> tuple[StageEvent, ...]:
    items = _reader.expect_array(value, "evidence.stage_events")
    events: list[StageEvent] = []
    seen: set[tuple[str, int, int]] = set()
    for index, item in enumerate(items):
        event = _read_stage_event(item, index, transcript, node_ids)
        key = (event.node_id, event.turn, event.start)
        if key in seen:
            fail(
                JudgeCode.INVALID_EVIDENCE,
                (
                    f"stage_events[{index}] repeats node {event.node_id} "
                    f"at turn {event.turn}, start {event.start}."
                ),
            )
        seen.add(key)
        events.append(event)
    return tuple(events)


def _require_turn_index(value: JsonValue, location: str, turn_count: int) -> int:
    number = _reader.require_int(value, location, label="turn index")
    if number < 0 or number >= turn_count:
        fail(JudgeCode.INVALID_EVIDENCE, f"{location} {number} is not a transcript turn.")
    return number


def _require_evidence_turns(value: JsonValue, location: str, turn_count: int) -> tuple[int, ...]:
    items = _reader.expect_array(value, location)
    indexes: list[int] = []
    seen: set[int] = set()
    for index, item in enumerate(items):
        turn_index = _require_turn_index(item, f"{location}[{index}]", turn_count)
        if turn_index in seen:
            fail(JudgeCode.INVALID_EVIDENCE, f"{location} repeats transcript turn {turn_index}.")
        seen.add(turn_index)
        indexes.append(turn_index)
    indexes.sort()
    return tuple(indexes)


def _applicability_from_name(name: str, location: str) -> Applicability:
    return enum_member(
        Applicability,
        name,
        location,
        _reject,
        supported="Supported values are applicable, not_applicable, and uncertain.",
    )


def _result_from_name(name: str, location: str) -> FindingResult:
    return enum_member(
        FindingResult,
        name,
        location,
        _reject,
        supported="Supported results are met, unmet, contradicted, and insufficient.",
    )


def _require_action_membership(
    action_turn: int | None,
    evidence_turns: tuple[int, ...],
    location: str,
) -> None:
    if action_turn is None:
        return
    if action_turn not in evidence_turns:
        fail(
            JudgeCode.INVALID_EVIDENCE,
            f"{location}.action_turn {action_turn} is not in evidence_turns.",
        )


def _require_result_shape(
    result: FindingResult,
    evidence_turns: tuple[int, ...],
    action_turn: int | None,
    location: str,
    turn_count: int,
) -> None:
    if result is FindingResult.MET or result is FindingResult.CONTRADICTED:
        if not evidence_turns:
            fail(JudgeCode.INVALID_EVIDENCE, f"{location} result {result} needs evidence turns.")
        if action_turn is None:
            fail(JudgeCode.INVALID_EVIDENCE, f"{location} result {result} needs an action turn.")
        return
    if action_turn is not None:
        fail(JudgeCode.INVALID_EVIDENCE, f"{location} result {result} cannot have an action turn.")
    if result is FindingResult.INSUFFICIENT and turn_count > 0 and not evidence_turns:
        fail(
            JudgeCode.INVALID_EVIDENCE,
            f"{location} is insufficient, so it must cite the ambiguous turns.",
        )


def _require_applicability_shape(finding: CriterionFinding, location: str) -> None:
    if finding.applicability is Applicability.NOT_APPLICABLE:
        if (
            finding.result is not FindingResult.UNMET
            or finding.evidence_turns
            or finding.action_turn is not None
        ):
            fail(
                JudgeCode.INVALID_EVIDENCE,
                f"{location} is not applicable, so result must be unmet and no turns can be cited.",
            )
        return
    if (
        finding.applicability is Applicability.UNCERTAIN
        and finding.result is not FindingResult.INSUFFICIENT
    ):
        fail(
            JudgeCode.INVALID_EVIDENCE, f"{location} is uncertain, so result must be insufficient."
        )


def _require_required_applicable(
    finding: CriterionFinding,
    obligation: _JudgedObligation,
    location: str,
) -> None:
    if obligation.requirement_type is not RequirementType.REQUIRED:
        return
    if finding.applicability is not Applicability.APPLICABLE:
        fail(
            JudgeCode.INVALID_EVIDENCE,
            f"{location} is required, so applicability must be applicable.",
        )


def _require_alternative_findings(
    findings: Mapping[str, CriterionFinding],
    obligations: tuple[_JudgedObligation, ...],
) -> None:
    grouped: dict[tuple[str, str], list[_JudgedObligation]] = {}
    for item in obligations:
        if (
            item.requirement_type is not RequirementType.ALTERNATIVE
            or item.alternative_group is None
        ):
            continue
        grouped.setdefault((_group_scope(item), item.alternative_group), []).append(item)
    for (scope, group), members in grouped.items():
        applicable = [
            member
            for member in members
            if findings[member.id].applicability is Applicability.APPLICABLE
        ]
        if len(applicable) > 1:
            fail(
                JudgeCode.INVALID_EVIDENCE,
                f"Alternative group {group!r} under {scope} has more than one applicable case.",
            )
        if len(applicable) != 1:
            continue
        chosen = applicable[0].id
        for member in members:
            if member.id == chosen:
                continue
            if findings[member.id].applicability is not Applicability.NOT_APPLICABLE:
                fail(
                    JudgeCode.INVALID_EVIDENCE,
                    (
                        f"Alternative group {group!r} under {scope} has an applicable case, "
                        f"so {member.id} must be not applicable."
                    ),
                )


def _require_parent_support(findings: Mapping[str, CriterionFinding], rubric: Rubric) -> None:
    for criterion in rubric.criteria:
        if not criterion.subrequirements:
            continue
        parent = findings[criterion.id]
        if parent.result is not FindingResult.MET:
            continue
        for subrequirement in criterion.subrequirements:
            child = findings[subrequirement.id]
            if child.applicability is not Applicability.APPLICABLE:
                continue
            if (
                subrequirement.requirement_type is RequirementType.REQUIRED
                and child.result is not FindingResult.MET
            ):
                fail(
                    JudgeCode.INVALID_EVIDENCE,
                    f"{criterion.id} is met, but required {subrequirement.id} is {child.result}.",
                )
            if (
                child.result is FindingResult.CONTRADICTED
                or child.result is FindingResult.INSUFFICIENT
            ):
                fail(
                    JudgeCode.INVALID_EVIDENCE,
                    f"{criterion.id} is met, but {subrequirement.id} is {child.result}.",
                )


def _read_finding(value: JsonValue, index: int, turn_count: int) -> CriterionFinding:
    location = f"findings[{index}]"
    fields = _reader.expect_exact_object(value, location, _FINDING_FIELDS)
    evidence_turns = _require_evidence_turns(
        _reader.require_field(fields, "evidence_turns", location),
        f"{location}.evidence_turns",
        turn_count,
    )
    action_value = _reader.require_field(fields, "action_turn", location)
    if action_value is None:
        action_turn = None
    else:
        action_turn = _require_turn_index(action_value, f"{location}.action_turn", turn_count)
    _require_action_membership(action_turn, evidence_turns, location)
    result = _result_from_name(
        _reader.require_text_field(fields, "result", location),
        f"{location}.result",
    )
    _require_result_shape(result, evidence_turns, action_turn, location, turn_count)
    applicability = _applicability_from_name(
        _reader.require_text_field(fields, "applicability", location),
        f"{location}.applicability",
    )
    finding = CriterionFinding(
        criterion_id=_reader.require_text_field(fields, "criterion_id", location),
        applicability=applicability,
        result=result,
        evidence_turns=evidence_turns,
        action_turn=action_turn,
        reason=_reader.require_text_field(fields, "reason", location),
    )
    _require_applicability_shape(finding, location)
    return finding


def _ordered_evidence(
    found: Mapping[str, CriterionFinding],
    obligations: tuple[_JudgedObligation, ...],
    stage_events: tuple[StageEvent, ...],
) -> Evidence:
    ordered = tuple(found[item.id] for item in obligations)
    return Evidence(
        findings=ordered,
        by_id=MappingProxyType({finding.criterion_id: finding for finding in ordered}),
        stage_events=stage_events,
    )


def parse_evidence(document: JsonValue, transcript: Transcript, rubric: Rubric) -> Evidence:
    """Validate decoded findings and ground them in the transcript and rubric."""

    code = JudgeCode.INVALID_EVIDENCE
    fields = _reader.expect_exact_object(document, "evidence", _EVIDENCE_FIELDS)
    items = _reader.expect_array(
        _reader.require_field(fields, "findings", "evidence"),
        "evidence.findings",
    )
    obligations = _judged_obligations(rubric)
    known = {item.id: item for item in obligations}
    found: dict[str, CriterionFinding] = {}
    seen: dict[str, str] = {}
    turn_count = len(transcript.turns)
    for index, item in enumerate(items):
        finding = _read_finding(item, index, turn_count)
        location = f"findings[{index}]"
        obligation = known.get(finding.criterion_id)
        if obligation is None:
            fail(code, f"{location}.criterion_id {finding.criterion_id!r} is not in the rubric.")
        previous = seen.get(finding.criterion_id)
        if previous is not None:
            fail(code, f"{location}.criterion_id {finding.criterion_id!r} duplicates {previous}.")
        seen[finding.criterion_id] = location
        _require_required_applicable(finding, obligation, location)
        found[finding.criterion_id] = finding
    missing = [item.id for item in obligations if item.id not in found]
    if missing:
        fail(code, f"Every obligation needs a finding. Missing: {', '.join(missing)}.")
    _require_alternative_findings(found, obligations)
    _require_parent_support(found, rubric)
    stage_events = _read_stage_events(
        _reader.require_field(fields, "stage_events", "evidence"),
        transcript,
        frozenset(rubric.by_node),
    )
    return _ordered_evidence(found, obligations, stage_events)


def _empty_finding(item: _JudgedObligation) -> CriterionFinding:
    if item.requirement_type is RequirementType.REQUIRED:
        return CriterionFinding(
            criterion_id=item.id,
            applicability=Applicability.APPLICABLE,
            result=FindingResult.UNMET,
            evidence_turns=(),
            action_turn=None,
            reason="The transcript has no turns, so the obligation was not met.",
        )
    if item.parent_id is None and item.requirement_type is RequirementType.CONDITIONAL:
        return CriterionFinding(
            criterion_id=item.id,
            applicability=Applicability.NOT_APPLICABLE,
            result=FindingResult.UNMET,
            evidence_turns=(),
            action_turn=None,
            reason="The transcript has no turns, so the condition did not occur.",
        )
    return CriterionFinding(
        criterion_id=item.id,
        applicability=Applicability.UNCERTAIN,
        result=FindingResult.INSUFFICIENT,
        evidence_turns=(),
        action_turn=None,
        reason="The transcript has no turns, so whether this obligation applies is unsettled.",
    )


def _evidence_without_turns(rubric: Rubric) -> Evidence:
    obligations = _judged_obligations(rubric)
    found: dict[str, CriterionFinding] = {}
    for item in obligations:
        finding = _empty_finding(item)
        _require_result_shape(
            finding.result,
            finding.evidence_turns,
            finding.action_turn,
            item.id,
            0,
        )
        _require_applicability_shape(finding, item.id)
        _require_required_applicable(finding, item, item.id)
        found[item.id] = finding
    _require_alternative_findings(found, obligations)
    _require_parent_support(found, rubric)
    return _ordered_evidence(found, obligations, ())


def _evidence_payload(transcript: Transcript, rubric: Rubric) -> dict[str, JsonValue]:
    turns: list[JsonValue] = []
    for turn in transcript.turns:
        spoken: dict[str, JsonValue] = {
            "index": turn.index,
            "role": turn.role,
            "content": turn.content,
        }
        turns.append(spoken)
    obligations: list[JsonValue] = []
    for item in _judged_obligations(rubric):
        obligation: dict[str, JsonValue] = {
            "id": item.id,
            "node_id": item.node_id,
            "parent_id": item.parent_id,
            "requirement_type": item.requirement_type.value,
            "condition": item.condition,
            "alternative_group": item.alternative_group,
            "exact_wording": item.exact_wording,
            "description": item.description,
            "expected_behavior": item.expected_behavior,
        }
        obligations.append(obligation)
    return {"turns": turns, "obligations": obligations}


def extract_evidence(
    transcript: Transcript,
    rubric: Rubric,
    completer: StructuredCompleter,
) -> Evidence:
    """Judge each rubric obligation against the transcript.

    A schema or grounding failure is sent back once for repair.
    An empty transcript is judged from the rubric alone, because there are no turns to cite.
    """

    if not transcript.turns:
        return _evidence_without_turns(rubric)
    payload = _evidence_payload(transcript, rubric)

    def repair_content(raw: str, exc: JudgeError) -> str:
        repair_payload: dict[str, JsonValue] = {
            "turns": payload["turns"],
            "obligations": payload["obligations"],
            "rejected_findings": raw,
            "validation_error": str(exc),
        }
        return json.dumps(repair_payload, ensure_ascii=False)

    return complete_or_repair(
        completer,
        instructions=_EVIDENCE_INSTRUCTIONS,
        repair_instructions=_EVIDENCE_REPAIR_INSTRUCTIONS,
        content=json.dumps(payload, ensure_ascii=False),
        repair_content=repair_content,
        schema_name="criterion_findings",
        schema=_FINDINGS_SCHEMA,
        label="Evidence judgment",
        invalid=JudgeCode.INVALID_EVIDENCE,
        parse=lambda raw: parse_evidence(
            decode_document(raw, label="Evidence", reject=_reject),
            transcript,
            rubric,
        ),
    )


__all__ = [
    "extract_evidence",
    "parse_evidence",
]
