"""Run the evaluator and shape demo responses."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import NoReturn

from src.errors import ContractError, JudgeError
from src.evaluate import evaluate
from src.report import render_report
from src.utils import JsonReader, JsonValue, decode_json

from .catalog import CatalogCase, read_fixture_text

_EVALUATION_ERROR = "evaluation_error"


def _reject(message: str) -> NoReturn:
    raise ValueError(message)


_reader = JsonReader(_reject)


def error_document(outcome: str, code: str, message: str) -> dict[str, JsonValue]:
    """Build the same error object the command-line tool writes."""

    return {"error": {"outcome": outcome, "code": code, "message": message}}


def case_list(cases: tuple[CatalogCase, ...]) -> dict[str, JsonValue]:
    """List every catalog case without evaluation output."""

    items: list[JsonValue] = []
    for case in cases:
        graph_present = case.graph_path is not None
        transcript_present = case.transcript_path is not None
        item: dict[str, JsonValue] = {
            "id": case.id,
            "title": case.title,
            "category": case.category,
            "graph_present": graph_present,
            "transcript_present": transcript_present,
            "runnable": graph_present and transcript_present,
            "notes": case.notes,
        }
        items.append(item)
    return {"cases": items}


def case_inputs(case: CatalogCase) -> dict[str, JsonValue]:
    """Return the raw graph and transcript text for one case."""

    return {
        "id": case.id,
        "title": case.title,
        "category": case.category,
        "notes": case.notes,
        "graph_text": read_fixture_text(case.graph_path),
        "transcript_text": read_fixture_text(case.transcript_path),
    }


def submission(document: JsonValue) -> tuple[str, str]:
    """Read graph and transcript text from a request body."""

    fields = _reader.expect_object(document, "body")
    graph = _reader.require_text_field(fields, "graph", "body", allow_empty=True)
    transcript = _reader.require_text_field(fields, "transcript", "body", allow_empty=True)
    return graph, transcript


def parse_body(payload: bytes) -> JsonValue:
    """Decode a UTF-8 JSON request body."""

    try:
        text = payload.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError("The request body is not valid UTF-8.") from exc
    try:
        return decode_json(text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"The request body is not valid JSON: {exc.msg}.") from exc


def run_paths(graph_path: Path, transcript_path: Path) -> dict[str, JsonValue]:
    """Evaluate two files and return a report or an error object."""

    try:
        report = evaluate(graph_path, transcript_path)
    except ContractError as exc:
        return error_document(exc.outcome.value, exc.code.value, str(exc))
    except JudgeError as exc:
        return error_document(_EVALUATION_ERROR, exc.code.value, str(exc))
    rendered = decode_json(render_report(report))
    if isinstance(rendered, dict):
        return rendered
    raise ValueError("The report is not a JSON object.")


def run_texts(graph_text: str, transcript_text: str) -> dict[str, JsonValue]:
    """Evaluate submitted text. Either side may be empty."""

    if graph_text == "":
        return error_document("input_error", "missing_required_field", "The graph is empty.")
    if transcript_text == "":
        return error_document(
            "input_error",
            "missing_required_field",
            "The transcript is empty.",
        )
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        graph_path = root / "graph.json"
        transcript_path = root / "transcript.json"
        graph_path.write_text(graph_text, encoding="utf-8")
        transcript_path.write_text(transcript_text, encoding="utf-8")
        return run_paths(graph_path, transcript_path)


def run_catalog(cases: tuple[CatalogCase, ...]) -> dict[str, JsonValue]:
    """Evaluate every case that has both files."""

    runs: list[JsonValue] = []
    for case in cases:
        graph_path = case.graph_path
        transcript_path = case.transcript_path
        if graph_path is None or transcript_path is None:
            continue
        entry: dict[str, JsonValue] = {
            "id": case.id,
            "title": case.title,
            "output": run_paths(graph_path, transcript_path),
        }
        runs.append(entry)
    return {"runs": runs}


__all__ = [
    "case_inputs",
    "case_list",
    "error_document",
    "parse_body",
    "run_catalog",
    "run_paths",
    "run_texts",
    "submission",
]
