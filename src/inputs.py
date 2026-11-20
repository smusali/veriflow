"""Load a workflow graph and a conversation transcript into typed models."""

from __future__ import annotations

import json
import math
from collections import deque
from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from types import MappingProxyType
from typing import Literal, NoReturn

type NodeType = Literal["start", "decision", "process", "end"]
type Role = Literal["user", "assistant"]


class Outcome(StrEnum):
    INPUT_ERROR = "input_error"
    UNSUPPORTED = "unsupported"


class ErrorCode(StrEnum):
    MALFORMED_JSON = "malformed_json"
    MISSING_REQUIRED_FIELD = "missing_required_field"
    UNSUPPORTED_ROLE = "unsupported_role"
    INVALID_TIMESTAMP = "invalid_timestamp"
    UNSORTED_TURNS = "unsorted_turns"
    DUPLICATE_NODE_ID = "duplicate_node_id"
    DANGLING_EDGE = "dangling_edge"
    UNSUPPORTED_GRAPH = "unsupported_graph"


class ContractError(Exception):
    """A workflow or transcript that cannot be evaluated."""

    outcome: Outcome
    code: ErrorCode

    def __init__(self, outcome: Outcome, code: ErrorCode, message: str) -> None:
        self.outcome = outcome
        self.code = code
        super().__init__(message)


@dataclass(frozen=True, slots=True)
class Position:
    x: int | float
    y: int | float


@dataclass(frozen=True, slots=True)
class NodeData:
    label: str
    description: str


@dataclass(frozen=True, slots=True)
class WorkflowNode:
    id: str
    type: NodeType
    data: NodeData
    position: Position | None


@dataclass(frozen=True, slots=True)
class WorkflowEdge:
    id: str
    source: str
    target: str


@dataclass(frozen=True, slots=True)
class Workflow:
    nodes: tuple[WorkflowNode, ...]
    edges: tuple[WorkflowEdge, ...]
    node_by_id: Mapping[str, WorkflowNode]
    outgoing: Mapping[str, tuple[str, ...]]
    start_id: str


@dataclass(frozen=True, slots=True)
class TranscriptTurn:
    """One transcript entry, with its zero-based array index."""

    index: int
    role: Role
    content: str
    beginning: int | float
    end: int | float


@dataclass(frozen=True, slots=True)
class Transcript:
    turns: tuple[TranscriptTurn, ...]


@dataclass(frozen=True, slots=True)
class _NodeParts:
    id: str
    type_name: str
    data: NodeData
    position: Position | None


def fail(code: ErrorCode, message: str) -> NoReturn:
    if code == ErrorCode.UNSUPPORTED_GRAPH:
        outcome = Outcome.UNSUPPORTED
    else:
        outcome = Outcome.INPUT_ERROR
    raise ContractError(outcome, code, message)


def reject_json_constant(constant: str) -> NoReturn:
    """Reject NaN and Infinity, which are not JSON numbers."""

    raise json.JSONDecodeError(f"invalid number {constant}", constant, 0)


def read_json_document(path: Path, label: str) -> object:
    try:
        text = path.read_text(encoding="utf-8")
    except UnicodeDecodeError as exc:
        fail(
            ErrorCode.MALFORMED_JSON,
            f"{label} file is not valid JSON: {exc.reason}.",
        )
    try:
        document: object = json.loads(text, parse_constant=reject_json_constant)
    except json.JSONDecodeError as exc:
        fail(
            ErrorCode.MALFORMED_JSON,
            f"{label} file is not valid JSON: {exc.msg}.",
        )
    return document


def expect_object(value: object, location: str) -> dict[str, object]:
    if not isinstance(value, dict):
        fail(ErrorCode.MISSING_REQUIRED_FIELD, f"{location} must be a JSON object.")
    collected: dict[str, object] = {}
    for key, item in value.items():
        if not isinstance(key, str):
            fail(ErrorCode.MISSING_REQUIRED_FIELD, f"{location} must use string keys.")
        collected[key] = item
    return collected


def expect_array(value: object, location: str) -> tuple[object, ...]:
    if not isinstance(value, list):
        fail(ErrorCode.MISSING_REQUIRED_FIELD, f"{location} must be a JSON array.")
    items: list[object] = []
    for item in value:
        items.append(item)
    return tuple(items)


def require_field(fields: Mapping[str, object], name: str, location: str) -> object:
    if name not in fields:
        fail(ErrorCode.MISSING_REQUIRED_FIELD, f"{location}.{name} is required.")
    return fields[name]


def require_text(value: object, location: str, *, allow_empty: bool) -> str:
    if not isinstance(value, str):
        fail(ErrorCode.MISSING_REQUIRED_FIELD, f"{location} must be a string.")
    if not allow_empty and value == "":
        fail(ErrorCode.MISSING_REQUIRED_FIELD, f"{location} must be a non-empty string.")
    return value


def require_text_field(
    fields: Mapping[str, object],
    name: str,
    location: str,
    *,
    allow_empty: bool,
) -> str:
    return require_text(
        require_field(fields, name, location),
        f"{location}.{name}",
        allow_empty=allow_empty,
    )


def read_finite_number(value: object, location: str, code: ErrorCode) -> int | float:
    # bool is an int subclass, so it is excluded before the int check.
    if isinstance(value, bool):
        fail(code, f"{location} must be a finite number.")
    if isinstance(value, int):
        return value
    if isinstance(value, float) and math.isfinite(value):
        return value
    fail(code, f"{location} must be a finite number.")


def require_timestamp(value: object, location: str) -> int | float:
    number = read_finite_number(value, location, ErrorCode.INVALID_TIMESTAMP)
    if number < 0:
        fail(ErrorCode.INVALID_TIMESTAMP, f"{location} must be nonnegative.")
    return number


def require_role(value: object, location: str) -> Role:
    if not isinstance(value, str):
        fail(ErrorCode.MISSING_REQUIRED_FIELD, f"{location} must be a string.")
    if value == "user":
        return "user"
    if value == "assistant":
        return "assistant"
    fail(
        ErrorCode.UNSUPPORTED_ROLE,
        f"{location} {value!r} is unsupported. Supported roles are user and assistant.",
    )


def node_type_from_name(type_name: str, location: str) -> NodeType:
    if type_name == "start":
        return "start"
    if type_name == "decision":
        return "decision"
    if type_name == "process":
        return "process"
    if type_name == "end":
        return "end"
    fail(
        ErrorCode.UNSUPPORTED_GRAPH,
        (
            f"{location} {type_name!r} is unsupported. "
            "Supported types are start, decision, process, and end."
        ),
    )


def read_position(value: object, location: str) -> Position:
    fields = expect_object(value, location)
    return Position(
        x=read_finite_number(
            require_field(fields, "x", location),
            f"{location}.x",
            ErrorCode.MISSING_REQUIRED_FIELD,
        ),
        y=read_finite_number(
            require_field(fields, "y", location),
            f"{location}.y",
            ErrorCode.MISSING_REQUIRED_FIELD,
        ),
    )


def read_node_parts(value: object, index: int) -> _NodeParts:
    location = f"nodes[{index}]"
    fields = expect_object(value, location)
    data_location = f"{location}.data"
    data_fields = expect_object(require_field(fields, "data", location), data_location)
    position = None
    if "position" in fields:
        position = read_position(fields["position"], f"{location}.position")
    return _NodeParts(
        id=require_text_field(fields, "id", location, allow_empty=False),
        type_name=require_text_field(fields, "type", location, allow_empty=True),
        data=NodeData(
            label=require_text_field(data_fields, "label", data_location, allow_empty=True),
            description=require_text_field(
                data_fields,
                "description",
                data_location,
                allow_empty=True,
            ),
        ),
        position=position,
    )


def finish_node(parts: _NodeParts, index: int) -> WorkflowNode:
    return WorkflowNode(
        id=parts.id,
        type=node_type_from_name(parts.type_name, f"nodes[{index}].type"),
        data=parts.data,
        position=parts.position,
    )


def read_edge(value: object, index: int) -> WorkflowEdge:
    location = f"edges[{index}]"
    fields = expect_object(value, location)
    return WorkflowEdge(
        id=require_text_field(fields, "id", location, allow_empty=False),
        source=require_text_field(fields, "source", location, allow_empty=False),
        target=require_text_field(fields, "target", location, allow_empty=False),
    )


def require_unique_ids(nodes: tuple[WorkflowNode, ...]) -> None:
    seen: dict[str, int] = {}
    for index, node in enumerate(nodes):
        previous = seen.get(node.id)
        if previous is not None:
            fail(
                ErrorCode.DUPLICATE_NODE_ID,
                f"nodes[{index}].id {node.id!r} duplicates nodes[{previous}].id.",
            )
        seen[node.id] = index


def require_endpoints(edges: tuple[WorkflowEdge, ...], node_ids: set[str]) -> None:
    for edge in edges:
        if edge.source not in node_ids:
            fail(
                ErrorCode.DANGLING_EDGE,
                f"Edge {edge.id!r} source {edge.source!r} does not match a node id.",
            )
        if edge.target not in node_ids:
            fail(
                ErrorCode.DANGLING_EDGE,
                f"Edge {edge.id!r} target {edge.target!r} does not match a node id.",
            )


def require_single_start(nodes: tuple[WorkflowNode, ...]) -> str:
    start_ids = tuple(node.id for node in nodes if node.type == "start")
    if len(start_ids) == 1:
        return start_ids[0]
    fail(
        ErrorCode.UNSUPPORTED_GRAPH,
        f"Workflow has {len(start_ids)} start nodes; exactly one is supported.",
    )


def build_outgoing(
    nodes: tuple[WorkflowNode, ...],
    edges: tuple[WorkflowEdge, ...],
) -> dict[str, tuple[str, ...]]:
    buckets: dict[str, list[str]] = {node.id: [] for node in nodes}
    for edge in edges:
        targets = buckets[edge.source]
        if edge.target not in targets:
            targets.append(edge.target)
    return {source: tuple(targets) for source, targets in buckets.items()}


def require_acyclic(outgoing: Mapping[str, tuple[str, ...]]) -> None:
    indegree = {node_id: 0 for node_id in outgoing}
    for targets in outgoing.values():
        for target in targets:
            indegree[target] += 1
    ready: deque[str] = deque(node_id for node_id, degree in indegree.items() if degree == 0)
    visited = 0
    while ready:
        node_id = ready.popleft()
        visited += 1
        for target in outgoing[node_id]:
            remaining = indegree[target] - 1
            indegree[target] = remaining
            if remaining == 0:
                ready.append(target)
    if visited != len(outgoing):
        fail(ErrorCode.UNSUPPORTED_GRAPH, "Workflow contains a cycle.")


def require_reachable_end(
    start_id: str,
    nodes: tuple[WorkflowNode, ...],
    outgoing: Mapping[str, tuple[str, ...]],
) -> None:
    types_by_id = {node.id: node.type for node in nodes}
    pending: deque[str] = deque((start_id,))
    seen: set[str] = set()
    while pending:
        current = pending.popleft()
        if current in seen:
            continue
        seen.add(current)
        if types_by_id[current] == "end":
            return
        pending.extend(outgoing[current])
    fail(
        ErrorCode.UNSUPPORTED_GRAPH,
        f"No end node is reachable from start {start_id!r}.",
    )


def parse_workflow(document: object) -> Workflow:
    """Validate a decoded workflow object and return its typed graph."""

    root = expect_object(document, "workflow")
    node_items = expect_array(require_field(root, "nodes", "workflow"), "workflow.nodes")
    edge_items = expect_array(require_field(root, "edges", "workflow"), "workflow.edges")
    parts = tuple(read_node_parts(item, index) for index, item in enumerate(node_items))
    edges = tuple(read_edge(item, index) for index, item in enumerate(edge_items))
    nodes = tuple(finish_node(part, index) for index, part in enumerate(parts))
    require_unique_ids(nodes)
    require_endpoints(edges, {node.id for node in nodes})
    start_id = require_single_start(nodes)
    outgoing = build_outgoing(nodes, edges)
    require_acyclic(outgoing)
    require_reachable_end(start_id, nodes, outgoing)
    return Workflow(
        nodes=nodes,
        edges=edges,
        node_by_id=MappingProxyType({node.id: node for node in nodes}),
        outgoing=MappingProxyType(outgoing),
        start_id=start_id,
    )


def read_turn(value: object, index: int) -> TranscriptTurn:
    location = f"turns[{index}]"
    fields = expect_object(value, location)
    role_value = require_field(fields, "role", location)
    content = require_text_field(fields, "content", location, allow_empty=True)
    beginning_value = require_field(fields, "beginning", location)
    end_value = require_field(fields, "end", location)
    role = require_role(role_value, f"{location}.role")
    beginning = require_timestamp(beginning_value, f"{location}.beginning")
    end = require_timestamp(end_value, f"{location}.end")
    if end < beginning:
        fail(
            ErrorCode.INVALID_TIMESTAMP,
            f"{location}.end {end} is earlier than beginning {beginning}.",
        )
    return TranscriptTurn(
        index=index,
        role=role,
        content=content,
        beginning=beginning,
        end=end,
    )


def parse_transcript(document: object) -> Transcript:
    """Validate a decoded transcript array and keep its original order."""

    items = expect_array(document, "transcript")
    turns: list[TranscriptTurn] = []
    previous_beginning: int | float | None = None
    for index, item in enumerate(items):
        turn = read_turn(item, index)
        if previous_beginning is not None and turn.beginning < previous_beginning:
            fail(
                ErrorCode.UNSORTED_TURNS,
                (
                    f"turns[{index}].beginning {turn.beginning} is earlier than "
                    f"the previous turn beginning {previous_beginning}."
                ),
            )
        previous_beginning = turn.beginning
        turns.append(turn)
    return Transcript(turns=tuple(turns))


def load_workflow(path: Path) -> Workflow:
    """Load a workflow graph from a JSON file."""

    return parse_workflow(read_json_document(path, "Graph"))


def load_transcript(path: Path) -> Transcript:
    """Load a transcript from a JSON file."""

    return parse_transcript(read_json_document(path, "Transcript"))


__all__ = [
    "ContractError",
    "ErrorCode",
    "NodeData",
    "NodeType",
    "Outcome",
    "Position",
    "Role",
    "Transcript",
    "TranscriptTurn",
    "Workflow",
    "WorkflowEdge",
    "WorkflowNode",
    "load_transcript",
    "load_workflow",
    "parse_transcript",
    "parse_workflow",
]
