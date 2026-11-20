"""Derive atomic evaluation criteria from workflow descriptions."""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType

from .errors import JudgeCode, JudgeError, fail_judge as fail
from .inputs import Workflow
from .models import Criterion, RequirementType, Rubric, StructuredCompleter, Subrequirement
from .prompts import RUBRIC_SCHEMA as _RUBRIC_SCHEMA
from .prompts import rubric_instructions
from .utils import JsonReader, JsonValue, decode_json

_SLUG = re.compile(r"[a-z][a-z0-9_]*\Z")
_EXACT_DEMAND = re.compile(
    r"\bexact\b|\bverbatim\b|\bword for word\b|\bmust say\b",
    re.IGNORECASE,
)
_SUBREQUIREMENT_FIELDS = frozenset(
    {
        "id",
        "description",
        "requirement_type",
        "condition",
        "alternative_group",
        "exact_wording",
        "expected_behavior",
        "source_text",
    }
)
_CRITERION_FIELDS = _SUBREQUIREMENT_FIELDS | {"node_id", "subrequirements"}
_RUBRIC_FIELDS = frozenset({"criteria"})


@dataclass(frozen=True, slots=True)
class _AlternativeMember:
    scope: str
    group: str
    condition: str


@dataclass(frozen=True, slots=True)
class _Obligation:
    id: str
    description: str
    requirement_type: RequirementType
    condition: str | None
    alternative_group: str | None
    exact_wording: bool
    expected_behavior: str
    source_text: str
    alternative: tuple[str, str] | None



_reader = JsonReader(
    lambda message: fail(JudgeCode.INVALID_RUBRIC, message),
    reject_whitespace=True,
)


def decode_document(
    text: str,
    *,
    code: JudgeCode = JudgeCode.INVALID_RUBRIC,
    label: str = "Rubric",
) -> JsonValue:
    try:
        return decode_json(text)
    except json.JSONDecodeError as exc:
        fail(code, f"{label} is not valid JSON: {exc.msg}.")


def expect_object(value: JsonValue, location: str) -> dict[str, JsonValue]:
    return _reader.expect_object(value, location)


def expect_exact_object(
    value: JsonValue,
    location: str,
    allowed: frozenset[str],
) -> dict[str, JsonValue]:
    return _reader.expect_exact_object(value, location, allowed)


def expect_array(value: JsonValue, location: str) -> tuple[JsonValue, ...]:
    return _reader.expect_array(value, location)


def require_field(
    fields: Mapping[str, JsonValue],
    name: str,
    location: str,
) -> JsonValue:
    return _reader.require_field(fields, name, location)


def require_text(value: JsonValue, location: str) -> str:
    return _reader.require_text(value, location)


def require_text_field(
    fields: Mapping[str, JsonValue],
    name: str,
    location: str,
) -> str:
    return _reader.require_text_field(fields, name, location)


def require_optional_text(value: JsonValue, location: str) -> str | None:
    return _reader.require_optional_text(value, location)


def require_bool(value: JsonValue, location: str) -> bool:
    return _reader.require_bool(value, location)


def requirement_type_from_name(name: str, location: str) -> RequirementType:
    if name == "required":
        return RequirementType.REQUIRED
    if name == "conditional":
        return RequirementType.CONDITIONAL
    if name == "alternative":
        return RequirementType.ALTERNATIVE
    fail(
        JudgeCode.INVALID_RUBRIC,
        (
            f"{location} {name!r} is unsupported. "
            "Supported types are required, conditional, and alternative."
        ),
    )


def normalize_text(value: str) -> str:
    return " ".join(value.casefold().split())


def checked_requirement(
    requirement_type: RequirementType,
    condition: str | None,
    alternative_group: str | None,
    location: str,
) -> tuple[str, str] | None:
    """Return an alternative's condition and group after checking the type rules."""

    if requirement_type is RequirementType.REQUIRED:
        if condition is not None or alternative_group is not None:
            fail(
                JudgeCode.INVALID_RUBRIC,
                f"{location} is required, so condition and alternative_group must be null.",
            )
        return None
    if condition is None:
        fail(JudgeCode.INVALID_RUBRIC, f"{location} needs a condition.")
    if requirement_type is RequirementType.CONDITIONAL:
        if alternative_group is not None:
            fail(
                JudgeCode.INVALID_RUBRIC,
                f"{location} is conditional, so alternative_group must be null.",
            )
        return None
    if alternative_group is None:
        fail(
            JudgeCode.INVALID_RUBRIC,
            f"{location} is alternative, so alternative_group must be set.",
        )
    return condition, alternative_group


def require_identifier(criterion_id: str, node_id: str, location: str) -> None:
    prefix = f"{node_id}."
    if not criterion_id.startswith(prefix):
        fail(
            JudgeCode.INVALID_RUBRIC,
            f"{location}.id must start with the node id {node_id!r} and a dot.",
        )
    slug = criterion_id[len(prefix) :]
    if _SLUG.fullmatch(slug) is None:
        fail(
            JudgeCode.INVALID_RUBRIC,
            (
                f"{location}.id slug must start with a lowercase letter and use only "
                "lowercase letters, digits, and underscores."
            ),
        )


def require_excerpt(source_text: str, description: str, location: str) -> None:
    excerpt = normalize_text(source_text)
    if excerpt not in normalize_text(description):
        fail(
            JudgeCode.INVALID_RUBRIC,
            f"{location}.source_text is not an excerpt of the node description.",
        )


def require_wording_flag(exact_wording: bool, description: str, location: str) -> None:
    if exact_wording and _EXACT_DEMAND.search(description) is None:
        fail(
            JudgeCode.INVALID_RUBRIC,
            f"{location} marks exact wording, but the node description does not demand it.",
        )


def claim_id(seen: dict[str, str], item_id: str, location: str) -> None:
    previous = seen.get(item_id)
    if previous is not None:
        fail(
            JudgeCode.INVALID_RUBRIC,
            f"{location}.id {item_id!r} duplicates {previous}.",
        )
    seen[item_id] = location


def require_alternative_groups(members: tuple[_AlternativeMember, ...]) -> None:
    grouped: dict[tuple[str, str], list[str]] = {}
    for member in members:
        grouped.setdefault((member.scope, member.group), []).append(member.condition)
    for (scope, group), conditions in grouped.items():
        if len(conditions) < 2:
            fail(
                JudgeCode.INVALID_RUBRIC,
                f"Alternative group {group!r} under {scope} needs at least two members.",
            )
        distinct = {normalize_text(condition) for condition in conditions}
        if len(distinct) < 2:
            fail(
                JudgeCode.INVALID_RUBRIC,
                f"Alternative group {group!r} under {scope} needs different conditions.",
            )


def read_obligation(
    fields: Mapping[str, JsonValue],
    location: str,
    node_id: str,
    node_description: str,
) -> _Obligation:
    requirement_type = requirement_type_from_name(
        require_text_field(fields, "requirement_type", location),
        f"{location}.requirement_type",
    )
    condition = require_optional_text(
        require_field(fields, "condition", location),
        f"{location}.condition",
    )
    alternative_group = require_optional_text(
        require_field(fields, "alternative_group", location),
        f"{location}.alternative_group",
    )
    item_id = require_text_field(fields, "id", location)
    require_identifier(item_id, node_id, location)
    source_text = require_text_field(fields, "source_text", location)
    require_excerpt(source_text, node_description, location)
    exact_wording = require_bool(
        require_field(fields, "exact_wording", location),
        f"{location}.exact_wording",
    )
    require_wording_flag(exact_wording, node_description, location)
    return _Obligation(
        id=item_id,
        description=require_text_field(fields, "description", location),
        requirement_type=requirement_type,
        condition=condition,
        alternative_group=alternative_group,
        exact_wording=exact_wording,
        expected_behavior=require_text_field(fields, "expected_behavior", location),
        source_text=source_text,
        alternative=checked_requirement(
            requirement_type,
            condition,
            alternative_group,
            location,
        ),
    )


def obligation_member(obligation: _Obligation, scope: str) -> _AlternativeMember | None:
    if obligation.alternative is None:
        return None
    condition, group = obligation.alternative
    return _AlternativeMember(scope=scope, group=group, condition=condition)


def read_subrequirement(
    value: JsonValue,
    index: int,
    parent: str,
    node_id: str,
    node_description: str,
    criterion_id: str,
) -> tuple[Subrequirement, _AlternativeMember | None]:
    location = f"{parent}.subrequirements[{index}]"
    fields = expect_exact_object(value, location, _SUBREQUIREMENT_FIELDS)
    obligation = read_obligation(fields, location, node_id, node_description)
    return (
        Subrequirement(
            id=obligation.id,
            description=obligation.description,
            requirement_type=obligation.requirement_type,
            condition=obligation.condition,
            alternative_group=obligation.alternative_group,
            exact_wording=obligation.exact_wording,
            expected_behavior=obligation.expected_behavior,
            source_text=obligation.source_text,
        ),
        obligation_member(obligation, f"criterion {criterion_id}"),
    )


def read_criterion(
    value: JsonValue,
    index: int,
    workflow: Workflow,
) -> tuple[Criterion, tuple[_AlternativeMember, ...]]:
    location = f"criteria[{index}]"
    fields = expect_exact_object(value, location, _CRITERION_FIELDS)
    node_id = require_text_field(fields, "node_id", location)
    node = workflow.node_by_id.get(node_id)
    if node is None:
        fail(
            JudgeCode.INVALID_RUBRIC,
            f"{location}.node_id {node_id!r} is not in the workflow.",
        )
    obligation = read_obligation(fields, location, node_id, node.data.description)
    raw_subrequirements = expect_array(
        require_field(fields, "subrequirements", location),
        f"{location}.subrequirements",
    )
    subrequirements: list[Subrequirement] = []
    members: list[_AlternativeMember] = []
    top_member = obligation_member(obligation, f"node {node_id}")
    if top_member is not None:
        members.append(top_member)
    for sub_index, item in enumerate(raw_subrequirements):
        subrequirement, member = read_subrequirement(
            item,
            sub_index,
            location,
            node_id,
            node.data.description,
            obligation.id,
        )
        subrequirements.append(subrequirement)
        if member is not None:
            members.append(member)
    return (
        Criterion(
            id=obligation.id,
            node_id=node_id,
            description=obligation.description,
            requirement_type=obligation.requirement_type,
            condition=obligation.condition,
            alternative_group=obligation.alternative_group,
            exact_wording=obligation.exact_wording,
            expected_behavior=obligation.expected_behavior,
            source_text=obligation.source_text,
            subrequirements=tuple(subrequirements),
        ),
        tuple(members),
    )


def order_criteria(criteria: tuple[Criterion, ...], workflow: Workflow) -> Rubric:
    grouped: dict[str, list[Criterion]] = {node.id: [] for node in workflow.nodes}
    for criterion in criteria:
        grouped[criterion.node_id].append(criterion)
    missing = [node.id for node in workflow.nodes if not grouped[node.id]]
    if missing:
        fail(
            JudgeCode.INVALID_RUBRIC,
            f"Every workflow node needs a criterion. Missing: {', '.join(missing)}.",
        )
    ordered: list[Criterion] = []
    by_node: dict[str, tuple[Criterion, ...]] = {}
    for node in workflow.nodes:
        items = tuple(grouped[node.id])
        by_node[node.id] = items
        ordered.extend(items)
    return Rubric(criteria=tuple(ordered), by_node=MappingProxyType(by_node))


_FOR_CASE = re.compile(r"(?:^|[.!?]\s+)For\s+([^:]+):\s*", re.IGNORECASE)
_SINGLE_QUOTED = re.compile(r":\s*'(.+?)'(?=\s|$)")
_DOUBLE_QUOTED = re.compile(r':\s*"(.+?)"(?=\s|$)')


def _sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    kept: list[str] = []
    for part in parts:
        cleaned = part.strip()
        if cleaned == "" or re.fullmatch(r"[A-Za-z]+:?", cleaned):
            continue
        kept.append(cleaned)
    return kept


def _quotes(text: str) -> list[str]:
    found = _SINGLE_QUOTED.findall(text)
    found.extend(_DOUBLE_QUOTED.findall(text))
    return [item.strip() for item in found if item.strip()]


def _outside_quotes(text: str) -> str:
    text = _SINGLE_QUOTED.sub(" ", text)
    return _DOUBLE_QUOTED.sub(" ", text)


def _first_sentence(text: str) -> str:
    match = re.search(r"[.!?]", text)
    if match is None:
        return text.strip()
    return text[: match.start()].strip()


def _detail_phrases(sentence: str) -> list[str]:
    text = re.sub(
        r"^(?:ask for|collect|obtain|get)\s+",
        "",
        sentence.strip(),
        flags=re.IGNORECASE,
    )
    text = re.sub(r"\s+to\s+[^.]+$", "", text, flags=re.IGNORECASE)
    phrases: list[str] = []
    for part in re.split(r",\s*|\s+and\s+", text):
        cleaned = re.sub(r"^(?:and|or)\s+", "", part.strip(" ."), flags=re.IGNORECASE)
        if cleaned:
            phrases.append(cleaned)
    return phrases


def _same_detail(left: str, right: str) -> bool:
    normalized_left = normalize_text(left)
    normalized_right = normalize_text(right)
    if normalized_left == normalized_right:
        return True
    left_words = re.findall(r"[A-Za-z]+", left)
    right_words = re.findall(r"[A-Za-z]+", right)
    left_initials = "".join(word[0] for word in left_words).casefold()
    right_initials = "".join(word[0] for word in right_words).casefold()
    return normalized_left == right_initials or normalized_right == left_initials


def _exclusive_cases(description: str) -> list[tuple[str, list[str]]]:
    matches = list(_FOR_CASE.finditer(description))
    if len(matches) < 2:
        return []
    cases: list[tuple[str, list[str]]] = []
    for index, match in enumerate(matches):
        end = matches[index + 1].start() if index + 1 < len(matches) else len(description)
        sentence = _first_sentence(description[match.end() : end])
        cases.append((match.group(1).strip(), _detail_phrases(sentence)))
    return cases


def _remainder_after_cases(description: str) -> str:
    matches = list(_FOR_CASE.finditer(description))
    if len(matches) < 2:
        return description
    body = description[matches[-1].end() :]
    sentence_end = re.search(r"[.!?]\s*", body)
    if sentence_end is None:
        return ""
    return body[sentence_end.end() :]


def _list_count(sentence: str) -> int:
    if sentence.strip() == "":
        return 0
    if re.search(r",\s*offer\b", sentence, re.IGNORECASE):
        return 2
    if sentence.count(",") >= 2:
        return len([part for part in sentence.split(",") if part.strip()])
    return 1


def _offer_confirm_count(sentence: str) -> int:
    if re.search(r"\boffer and confirm\b", sentence, re.IGNORECASE) is None:
        return 0
    count = 3
    if re.search(r"\bprepar", sentence, re.IGNORECASE):
        count += 1
    return count


def _minimum_criteria(description: str) -> int:
    quotes = _quotes(description)
    if len(quotes) >= 2:
        return len(quotes)
    minimum = 0
    text = description
    for quote in quotes:
        inner = _sentences(quote)
        if len(inner) >= 2:
            minimum += len(inner)
            text = _outside_quotes(description)
    if _exclusive_cases(description):
        minimum += 1
        text = _remainder_after_cases(description)
    for sentence in _sentences(text):
        offer_count = _offer_confirm_count(sentence)
        if offer_count:
            minimum += offer_count
            continue
        minimum += _list_count(sentence)
    return minimum


def _alternative_subs(criterion: Criterion) -> list[Subrequirement]:
    return [
        sub
        for sub in criterion.subrequirements
        if sub.requirement_type is RequirementType.ALTERNATIVE and sub.alternative_group is not None
    ]


def _source_inside(source: str, container: str) -> bool:
    return normalize_text(source) in normalize_text(container)


def node_split_problems(node_id: str, description: str, criteria: tuple[Criterion, ...]) -> list[str]:
    problems: list[str] = []
    minimum = _minimum_criteria(description)
    if len(criteria) < minimum:
        problems.append(
            f"Node {node_id} needs at least {minimum} top-level criteria and has {len(criteria)}. "
            "Each independently observable action is its own criterion."
        )
    compound = minimum > 1 or len(_exclusive_cases(description)) >= 2
    for criterion in criteria:
        if compound and normalize_text(criterion.source_text) == normalize_text(description):
            problems.append(
                f"{criterion.id} cites the entire node {node_id} description. "
                "Cite only the span for that obligation."
            )
        if criterion.subrequirements and not _alternative_subs(criterion):
            problems.append(
                f"{criterion.id} uses subrequirements without alternative cases. "
                "Subrequirements belong only under one required aggregate whose exclusive cases "
                "are alternative subrequirements. Other actions are top-level criteria."
            )
    cases = _exclusive_cases(description)
    if cases:
        aggregates = [criterion for criterion in criteria if _alternative_subs(criterion)]
        if not aggregates:
            labels = ", ".join(label for label, _phrases in cases)
            problems.append(
                f"Node {node_id} has exclusive cases ({labels}). "
                "Use one required aggregate criterion. Its case labels are alternative "
                "subrequirements in one group. Details listed for every case are required "
                "subrequirements. Details listed for one case are conditional subrequirements."
            )
        else:
            subs = aggregates[0].subrequirements
            problems.extend(_case_detail_problems(node_id, cases, subs))
    quotes = _quotes(description)
    if len(quotes) >= 2:
        conditional = [
            criterion
            for criterion in criteria
            if criterion.requirement_type is RequirementType.CONDITIONAL
            and not criterion.subrequirements
            and not criterion.exact_wording
        ]
        if len(conditional) < len(quotes):
            problems.append(
                f"Node {node_id} has {len(quotes)} quoted examples of separate situations. "
                "Each situation is its own top-level conditional criterion with exact_wording false. "
                "Do not place them under a required parent."
            )
    if len(quotes) == 1:
        quote = quotes[0]
        inner = _sentences(quote)
        if len(inner) >= 2:
            covering = [
                criterion
                for criterion in criteria
                if not criterion.exact_wording and _source_inside(criterion.source_text, quote)
            ]
            if len(covering) < len(inner):
                problems.append(
                    f"Node {node_id} has a quoted example with {len(inner)} sentences. "
                    "Each sentence is its own top-level criterion with exact_wording false."
                )
    for sentence in _sentences(description):
        expected = _offer_confirm_count(sentence)
        if expected == 0:
            continue
        covering = [
            criterion
            for criterion in criteria
            if criterion.requirement_type is RequirementType.REQUIRED
            and not criterion.subrequirements
            and _source_inside(criterion.source_text, sentence)
        ]
        if len(covering) < expected:
            problems.append(
                "Offer and confirm is three top-level criteria: the offer, the other party's "
                "acceptance, and the speaker's confirmation. "
                "A preparation instruction in that same sentence is a fourth criterion. "
                f"Node {node_id} has {len(covering)} of these and needs {expected}."
            )
    return problems


def _detail_clusters(
    cases: list[tuple[str, list[str]]],
) -> list[tuple[list[str], set[int]]]:
    clusters: list[tuple[list[str], set[int]]] = []
    items: list[tuple[int, str]] = []
    for index, (_label, phrases) in enumerate(cases):
        for phrase in phrases:
            items.append((index, phrase))
    used = [False] * len(items)
    for index, (case_index, phrase) in enumerate(items):
        if used[index]:
            continue
        phrases = [phrase]
        present = {case_index}
        used[index] = True
        for other_index in range(index + 1, len(items)):
            if used[other_index]:
                continue
            other_case, other_phrase = items[other_index]
            if any(_same_detail(other_phrase, existing) for existing in phrases):
                used[other_index] = True
                phrases.append(other_phrase)
                present.add(other_case)
        clusters.append((phrases, present))
    return clusters


def _counted_text(description: str) -> str:
    quotes = _quotes(description)
    if len(quotes) >= 2:
        return ""
    text = description
    for quote in quotes:
        if len(_sentences(quote)) >= 2:
            text = _outside_quotes(description)
    if _exclusive_cases(description):
        text = _remainder_after_cases(description)
    return text


def _shape_lines(node_id: str, description: str) -> list[str]:
    lines = [f"Node {node_id}."]
    quotes = _quotes(description)
    cases = _exclusive_cases(description)
    if len(quotes) >= 2:
        lines.append(
            f"Emit {len(quotes)} top-level conditional criteria, one for each quoted situation. "
            "exact_wording is false. Leave subrequirements empty. Do not add a required parent."
        )
    else:
        if len(quotes) == 1 and len(_sentences(quotes[0])) >= 2:
            sentence_count = len(_sentences(quotes[0]))
            lines.append(
                f"The quoted example has {sentence_count} sentences. "
                "Each sentence is its own top-level criterion with exact_wording false "
                "and empty subrequirements."
            )
        if cases:
            labels = ", ".join(label for label, _phrases in cases)
            lines.append(
                f"Emit one required aggregate for the exclusive cases ({labels}). "
                "That aggregate is invalid unless it has one alternative subrequirement per case, "
                "all in one alternative_group, each citing its own For span."
            )
            for phrases, present in _detail_clusters(cases):
                shown = " / ".join(phrases)
                if len(present) == len(cases):
                    lines.append(
                        f"On that same aggregate, add one required subrequirement for {shown}."
                    )
                else:
                    lines.append(
                        "On that same aggregate, add one conditional subrequirement for "
                        f"{shown}."
                    )
            lines.append("Do not put later actions inside the aggregate.")
        for sentence in _sentences(_counted_text(description)):
            offer_count = _offer_confirm_count(sentence)
            if offer_count:
                preparation = ""
                if offer_count > 3:
                    preparation = " The preparation instruction is the extra criterion."
                lines.append(
                    f"This sentence is {offer_count} required top-level criteria with empty "
                    "subrequirements: the offer, the other party's acceptance, and the "
                    f"speaker's confirmation.{preparation} Sentence: {sentence}"
                )
                continue
            count = _list_count(sentence)
            if count > 1:
                lines.append(
                    f"Split this sentence into {count} required top-level criteria with empty "
                    f"subrequirements: {sentence}"
                )
    if cases:
        lines.append(
            "Leave subrequirements empty on every criterion except that one aggregate."
        )
    else:
        lines.append("Leave every subrequirements array empty.")
    lines.append(
        f"Node {node_id} needs at least {_minimum_criteria(description)} top-level criteria."
    )
    return lines


def binding_shapes(workflow: Workflow) -> str:
    """State the criterion split the description checker requires."""

    blocks = [
        " ".join(_shape_lines(node.id, node.data.description)) for node in workflow.nodes
    ]
    return "\n".join(blocks)


def _case_detail_problems(
    node_id: str,
    cases: list[tuple[str, list[str]]],
    subs: tuple[Subrequirement, ...],
) -> list[str]:
    problems: list[str] = []
    for phrases, present in _detail_clusters(cases):
        shared = len(present) == len(cases)
        wanted = RequirementType.REQUIRED if shared else RequirementType.CONDITIONAL
        matched = [
            sub
            for sub in subs
            if sub.requirement_type is wanted
            and any(
                _source_inside(sub.source_text, phrase) or _source_inside(phrase, sub.source_text)
                for phrase in phrases
            )
        ]
        if matched:
            continue
        kind = "required" if shared else "conditional"
        shown = " / ".join(phrases)
        problems.append(
            f"Node {node_id} must include a {kind} subrequirement for {shown} "
            "under the aggregate case criterion."
        )
    return problems


def require_granularity(criteria: tuple[Criterion, ...], workflow: Workflow) -> None:
    grouped: dict[str, list[Criterion]] = {node.id: [] for node in workflow.nodes}
    for criterion in criteria:
        grouped[criterion.node_id].append(criterion)
    problems: list[str] = []
    for node in workflow.nodes:
        problems.extend(
            node_split_problems(node.id, node.data.description, tuple(grouped[node.id]))
        )
    if problems:
        fail(JudgeCode.INVALID_RUBRIC, "\n".join(problems))


def parse_rubric(document: JsonValue, workflow: Workflow) -> Rubric:
    """Validate a decoded rubric and ground it in the workflow descriptions."""

    fields = expect_exact_object(document, "rubric", _RUBRIC_FIELDS)
    items = expect_array(require_field(fields, "criteria", "rubric"), "rubric.criteria")
    criteria: list[Criterion] = []
    members: list[_AlternativeMember] = []
    seen: dict[str, str] = {}
    for index, item in enumerate(items):
        criterion, criterion_members = read_criterion(item, index, workflow)
        claim_id(seen, criterion.id, f"criteria[{index}]")
        for sub_index, subrequirement in enumerate(criterion.subrequirements):
            claim_id(
                seen,
                subrequirement.id,
                f"criteria[{index}].subrequirements[{sub_index}]",
            )
        criteria.append(criterion)
        members.extend(criterion_members)
    require_alternative_groups(tuple(members))
    require_granularity(tuple(criteria), workflow)
    return order_criteria(tuple(criteria), workflow)


def workflow_payload(workflow: Workflow) -> dict[str, JsonValue]:
    """Describe each node by id, label, and description."""

    nodes: list[dict[str, str]] = []
    for node in workflow.nodes:
        nodes.append(
            {
                "id": node.id,
                "label": node.data.label,
                "description": node.data.description,
            }
        )
    return {"nodes": nodes}


def build_rubric(workflow: Workflow, completer: StructuredCompleter) -> Rubric:
    """Derive one rubric from the workflow descriptions.

    A schema or grounding failure is sent back once for repair.
    """

    payload = workflow_payload(workflow)
    raw = completer.complete(
        instructions=rubric_instructions(binding_shapes(workflow), repair=False),
        content=json.dumps(payload, ensure_ascii=False),
        schema_name="workflow_rubric",
        schema=_RUBRIC_SCHEMA,
        label="Rubric generation",
    )
    try:
        return parse_rubric(decode_document(raw), workflow)
    except JudgeError as exc:
        if exc.code is not JudgeCode.INVALID_RUBRIC:
            raise
        repair_payload: dict[str, JsonValue] = {
            "nodes": payload["nodes"],
            "rejected_rubric": raw,
            "validation_errors": [line for line in str(exc).splitlines() if line.strip()],
        }
        repaired = completer.complete(
            instructions=rubric_instructions(binding_shapes(workflow), repair=True),
            content=json.dumps(repair_payload, ensure_ascii=False),
            schema_name="workflow_rubric",
            schema=_RUBRIC_SCHEMA,
            label="Rubric generation",
        )
        return parse_rubric(decode_document(repaired), workflow)


__all__ = [
    "binding_shapes",
    "build_rubric",
    "parse_rubric",
]
