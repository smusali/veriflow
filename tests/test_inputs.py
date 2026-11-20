"""Workflow and transcript parsing contracts."""

from __future__ import annotations

import re
from collections.abc import Callable
from pathlib import Path
from types import MappingProxyType

import pytest

from src.errors import ContractError, ErrorCode, Outcome
from src.inputs import (
    Position,
    build_incoming,
    load_transcript,
    load_workflow,
    parse_transcript,
    parse_workflow,
)
from src.utils import JsonValue

ROOT = Path(__file__).resolve().parents[1]


def node(
    node_id: str,
    node_type: str,
    *,
    label: str | None = None,
    description: str = "",
    data: JsonValue | None = None,
    position: JsonValue | None = None,
    include_id: bool = True,
    include_data: bool = True,
) -> dict[str, JsonValue]:
    payload: dict[str, JsonValue] = {"type": node_type}
    if include_id:
        payload["id"] = node_id
    if data is not None:
        payload["data"] = data
    elif include_data:
        payload["data"] = {
            "label": node_id if label is None else label,
            "description": description,
        }
    if position is not None:
        payload["position"] = position
    return payload


def edge(edge_id: str, source: str, target: str) -> dict[str, JsonValue]:
    return {"id": edge_id, "source": source, "target": target}


def without(fields: dict[str, JsonValue], key: str) -> dict[str, JsonValue]:
    del fields[key]
    return fields


def point(x: JsonValue, y: JsonValue) -> dict[str, JsonValue]:
    return {"x": x, "y": y}


def workflow(
    nodes: tuple[JsonValue, ...],
    edges: tuple[JsonValue, ...] = (),
) -> dict[str, JsonValue]:
    return {"nodes": list(nodes), "edges": list(edges)}


def turn(
    role: JsonValue,
    content: JsonValue,
    beginning: JsonValue,
    end: JsonValue,
) -> dict[str, JsonValue]:
    return {"role": role, "content": content, "beginning": beginning, "end": end}


def assert_read_only(mapping: object, key: str, value: object) -> None:
    """Read-only mappings reject item assignment at runtime."""

    with pytest.raises(TypeError):
        # The type has no item setter; assignment is the runtime contract.
        mapping[key] = value  # type: ignore[index]


def assert_contract(error: ContractError, code: ErrorCode, fragment: str) -> None:
    assert error.code is code
    expected = Outcome.UNSUPPORTED if code is ErrorCode.UNSUPPORTED_GRAPH else Outcome.INPUT_ERROR
    assert error.outcome is expected
    assert re.search(re.escape(fragment), str(error))


def expect_failure(
    load: Callable[[JsonValue], object],
    document: JsonValue,
    code: ErrorCode,
    fragment: str,
) -> None:
    with pytest.raises(ContractError) as caught:
        load(document)
    assert_contract(caught.value, code, fragment)


def test_parse_workflow_keeps_graph_structure() -> None:
    parsed = parse_workflow(
        workflow(
            (
                node("s", "start", label="", description=""),
                node("d", "decision"),
                node("p", "process", position=point(1.5, 2)),
                node("e", "end"),
            ),
            (
                edge("sd", "s", "d"),
                edge("dp", "d", "p"),
                edge("pe", "p", "e"),
                edge("pe-again", "p", "e"),
            ),
        )
    )
    assert [item.type for item in parsed.nodes] == ["start", "decision", "process", "end"]
    assert parsed.start_id == "s"
    assert parsed.nodes[0].data.label == ""
    assert parsed.nodes[0].data.description == ""
    assert parsed.nodes[0].position is None
    assert parsed.nodes[3].position is None
    assert parsed.nodes[2].position == Position(x=1.5, y=2)
    assert [item.id for item in parsed.edges] == ["sd", "dp", "pe", "pe-again"]
    assert parsed.outgoing["p"] == ("e",)
    assert parsed.outgoing["e"] == ()
    incoming = build_incoming(parsed)
    assert incoming["e"] == ("p",)
    assert incoming["d"] == ("s",)
    assert isinstance(parsed.node_by_id, MappingProxyType)
    assert isinstance(parsed.outgoing, MappingProxyType)
    assert_read_only(parsed.node_by_id, "extra", parsed.nodes[0])
    assert_read_only(parsed.outgoing, "s", ())


def test_parse_transcript_preserves_order_and_overlap() -> None:
    turns: list[JsonValue] = [
        turn("user", "", 0, 5),
        turn("assistant", "ready", 0, 1),
        turn("user", "thanks", 1, 1.25),
    ]
    parsed = parse_transcript(turns)
    assert [
        (item.index, item.role, item.content, item.beginning, item.end) for item in parsed.turns
    ] == [
        (0, "user", "", 0, 5),
        (1, "assistant", "ready", 0, 1),
        (2, "user", "thanks", 1, 1.25),
    ]


def test_parse_transcript_allows_an_empty_array() -> None:
    empty: JsonValue = []
    assert parse_transcript(empty).turns == ()


def test_assignment_fixtures_load() -> None:
    parsed = load_workflow(ROOT / "fixtures/assignment_graph.json")
    spoken = load_transcript(ROOT / "fixtures/assignment_transcript.json")
    assert len(parsed.nodes) == 5
    assert parsed.start_id == "1"
    assert len(spoken.turns) == 24


def test_load_round_trip(tmp_path: Path) -> None:
    graph_path = tmp_path / "graph.json"
    transcript_path = tmp_path / "transcript.json"
    graph_path.write_text(
        '{"nodes": [{"id": "s", "type": "start", "data": {"label": "S", "description": ""}},'
        '{"id": "e", "type": "end", "data": {"label": "E", "description": ""}}],'
        '"edges": [{"id": "se", "source": "s", "target": "e"}]}',
        encoding="utf-8",
    )
    transcript_path.write_text(
        '[{"role": "user", "content": "hi", "beginning": 0, "end": 1}]',
        encoding="utf-8",
    )
    assert load_workflow(graph_path).start_id == "s"
    assert load_transcript(transcript_path).turns[0].content == "hi"


@pytest.mark.parametrize(
    ("document", "code", "fragment"),
    [
        pytest.param(
            [],
            ErrorCode.MISSING_REQUIRED_FIELD,
            "workflow must be a JSON object.",
            id="root",
        ),
        pytest.param(
            {"edges": []},
            ErrorCode.MISSING_REQUIRED_FIELD,
            "workflow.nodes is required.",
            id="missing-nodes",
        ),
        pytest.param(
            {"nodes": []},
            ErrorCode.MISSING_REQUIRED_FIELD,
            "workflow.edges is required.",
            id="missing-edges",
        ),
        pytest.param(
            {"nodes": {}, "edges": []},
            ErrorCode.MISSING_REQUIRED_FIELD,
            "workflow.nodes must be a JSON array.",
            id="nodes-not-array",
        ),
        pytest.param(
            {"nodes": [], "edges": {}},
            ErrorCode.MISSING_REQUIRED_FIELD,
            "workflow.edges must be a JSON array.",
            id="edges-not-array",
        ),
        pytest.param(
            workflow(("no",), ()),
            ErrorCode.MISSING_REQUIRED_FIELD,
            "nodes[0] must be a JSON object.",
            id="node-not-object",
        ),
        pytest.param(
            workflow((node("s", "start"),), ("no",)),
            ErrorCode.MISSING_REQUIRED_FIELD,
            "edges[0] must be a JSON object.",
            id="edge-not-object",
        ),
        pytest.param(
            workflow((node("s", "start", include_id=False),), ()),
            ErrorCode.MISSING_REQUIRED_FIELD,
            "nodes[0].id is required.",
            id="missing-id",
        ),
        pytest.param(
            workflow((node("", "start"),), ()),
            ErrorCode.MISSING_REQUIRED_FIELD,
            "nodes[0].id must be a non-empty string.",
            id="empty-id",
        ),
        pytest.param(
            workflow((node("s", "start", include_data=False),), ()),
            ErrorCode.MISSING_REQUIRED_FIELD,
            "nodes[0].data is required.",
            id="missing-data",
        ),
        pytest.param(
            workflow((node("s", "start", data=[]),), ()),
            ErrorCode.MISSING_REQUIRED_FIELD,
            "nodes[0].data must be a JSON object.",
            id="data-not-object",
        ),
        pytest.param(
            workflow((node("s", "start", data={"description": ""}),), ()),
            ErrorCode.MISSING_REQUIRED_FIELD,
            "nodes[0].data.label is required.",
            id="missing-label",
        ),
        pytest.param(
            workflow((node("s", "start", data={"label": "S"}),), ()),
            ErrorCode.MISSING_REQUIRED_FIELD,
            "nodes[0].data.description is required.",
            id="missing-description",
        ),
        pytest.param(
            workflow(
                (node("s", "start"), node("e", "end")),
                (without(edge("e1", "s", "e"), "source"),),
            ),
            ErrorCode.MISSING_REQUIRED_FIELD,
            "edges[0].source is required.",
            id="missing-source",
        ),
        pytest.param(
            workflow(
                (node("s", "start"), node("e", "end")),
                (without(edge("e1", "s", "e"), "target"),),
            ),
            ErrorCode.MISSING_REQUIRED_FIELD,
            "edges[0].target is required.",
            id="missing-target",
        ),
        pytest.param(
            workflow((node("s", "fork"),), ()),
            ErrorCode.UNSUPPORTED_GRAPH,
            "nodes[0].type 'fork' is unsupported.",
            id="unsupported-type",
        ),
        pytest.param(
            workflow((node("s", "start", position=[]),), ()),
            ErrorCode.MISSING_REQUIRED_FIELD,
            "nodes[0].position must be a JSON object.",
            id="position-not-object",
        ),
        pytest.param(
            workflow((node("s", "start", position=point(True, 0)),), ()),
            ErrorCode.MISSING_REQUIRED_FIELD,
            "nodes[0].position.x must be a finite number.",
            id="position-bool",
        ),
        pytest.param(
            workflow((node("1", "start"), node("1", "end")), ()),
            ErrorCode.DUPLICATE_NODE_ID,
            "nodes[1].id '1' duplicates nodes[0].id.",
            id="duplicate-id",
        ),
        pytest.param(
            workflow(
                (node("s", "start"), node("e", "end")),
                (edge("e1", "missing", "e"),),
            ),
            ErrorCode.DANGLING_EDGE,
            "Edge 'e1' source 'missing' does not match a node id.",
            id="dangling-source",
        ),
        pytest.param(
            workflow(
                (node("s", "start"), node("e", "end")),
                (edge("e1", "s", "missing"),),
            ),
            ErrorCode.DANGLING_EDGE,
            "Edge 'e1' target 'missing' does not match a node id.",
            id="dangling-target",
        ),
        pytest.param(
            workflow((node("e", "end"),), ()),
            ErrorCode.UNSUPPORTED_GRAPH,
            "Workflow has 0 start nodes; exactly one is supported.",
            id="no-start",
        ),
        pytest.param(
            workflow(
                (node("a", "start"), node("b", "start"), node("e", "end")),
                (edge("ae", "a", "e"), edge("be", "b", "e")),
            ),
            ErrorCode.UNSUPPORTED_GRAPH,
            "Workflow has 2 start nodes; exactly one is supported.",
            id="two-starts",
        ),
        pytest.param(
            workflow(
                (node("s", "start"), node("e", "end")),
                (edge("se", "s", "e"), edge("es", "e", "s")),
            ),
            ErrorCode.UNSUPPORTED_GRAPH,
            "Workflow contains a cycle.",
            id="cycle",
        ),
        pytest.param(
            workflow((node("s", "start"), node("e", "end")), ()),
            ErrorCode.UNSUPPORTED_GRAPH,
            "No end node is reachable from start 's'.",
            id="unreachable-end",
        ),
    ],
)
def test_parse_workflow_rejects(document: JsonValue, code: ErrorCode, fragment: str) -> None:
    expect_failure(parse_workflow, document, code, fragment)


@pytest.mark.parametrize(
    ("document", "code", "fragment"),
    [
        pytest.param(
            {"turns": []},
            ErrorCode.MISSING_REQUIRED_FIELD,
            "transcript must be a JSON array.",
            id="root",
        ),
        pytest.param(
            ["no"],
            ErrorCode.MISSING_REQUIRED_FIELD,
            "turns[0] must be a JSON object.",
            id="turn-not-object",
        ),
        pytest.param(
            [without(turn("user", "hi", 0, 1), "role")],
            ErrorCode.MISSING_REQUIRED_FIELD,
            "turns[0].role is required.",
            id="missing-role",
        ),
        pytest.param(
            [turn(1, "hi", 0, 1)],
            ErrorCode.MISSING_REQUIRED_FIELD,
            "turns[0].role must be a string.",
            id="role-not-string",
        ),
        pytest.param(
            [turn("system", "hi", 0, 1)],
            ErrorCode.UNSUPPORTED_ROLE,
            "turns[0].role 'system' is unsupported.",
            id="unsupported-role",
        ),
        pytest.param(
            [without(turn("user", "hi", 0, 1), "content")],
            ErrorCode.MISSING_REQUIRED_FIELD,
            "turns[0].content is required.",
            id="missing-content",
        ),
        pytest.param(
            [turn("user", 1, 0, 1)],
            ErrorCode.MISSING_REQUIRED_FIELD,
            "turns[0].content must be a string.",
            id="content-not-string",
        ),
        pytest.param(
            [without(turn("user", "hi", 0, 1), "beginning")],
            ErrorCode.MISSING_REQUIRED_FIELD,
            "turns[0].beginning is required.",
            id="missing-beginning",
        ),
        pytest.param(
            [without(turn("user", "hi", 0, 1), "end")],
            ErrorCode.MISSING_REQUIRED_FIELD,
            "turns[0].end is required.",
            id="missing-end",
        ),
        pytest.param(
            [turn("user", "hi", -1, 0)],
            ErrorCode.INVALID_TIMESTAMP,
            "turns[0].beginning must be nonnegative.",
            id="negative-time",
        ),
        pytest.param(
            [turn("user", "hi", True, 1)],
            ErrorCode.INVALID_TIMESTAMP,
            "turns[0].beginning must be a finite number.",
            id="bool-time",
        ),
        pytest.param(
            [turn("user", "hi", "0", 1)],
            ErrorCode.INVALID_TIMESTAMP,
            "turns[0].beginning must be a finite number.",
            id="string-time",
        ),
        pytest.param(
            [turn("user", "hi", 2, 1)],
            ErrorCode.INVALID_TIMESTAMP,
            "turns[0].end 1 is earlier than beginning 2.",
            id="end-before-beginning",
        ),
        pytest.param(
            [turn("user", "hi", 1, 2), turn("assistant", "ok", 0, 1)],
            ErrorCode.UNSORTED_TURNS,
            "turns[1].beginning 0 is earlier than the previous turn beginning 1.",
            id="unsorted",
        ),
    ],
)
def test_parse_transcript_rejects(document: JsonValue, code: ErrorCode, fragment: str) -> None:
    expect_failure(parse_transcript, document, code, fragment)


@pytest.mark.parametrize(
    ("loader", "filename", "fragment"),
    [
        pytest.param(load_workflow, "graph.json", "Graph file is not valid JSON", id="graph"),
        pytest.param(
            load_transcript,
            "transcript.json",
            "Transcript file is not valid JSON",
            id="transcript",
        ),
    ],
)
def test_load_rejects_malformed_json(
    tmp_path: Path,
    loader: Callable[[Path], object],
    filename: str,
    fragment: str,
) -> None:
    path = tmp_path / filename
    path.write_text("{", encoding="utf-8")
    with pytest.raises(ContractError) as caught:
        loader(path)
    assert_contract(caught.value, ErrorCode.MALFORMED_JSON, fragment)


@pytest.mark.parametrize(
    ("loader", "filename", "fragment"),
    [
        pytest.param(load_workflow, "graph.json", "Graph file is not valid JSON", id="graph"),
        pytest.param(
            load_transcript,
            "transcript.json",
            "Transcript file is not valid JSON",
            id="transcript",
        ),
    ],
)
def test_load_rejects_non_utf8(
    tmp_path: Path,
    loader: Callable[[Path], object],
    filename: str,
    fragment: str,
) -> None:
    path = tmp_path / filename
    path.write_bytes(b"\xff")
    with pytest.raises(ContractError) as caught:
        loader(path)
    assert_contract(caught.value, ErrorCode.MALFORMED_JSON, fragment)
