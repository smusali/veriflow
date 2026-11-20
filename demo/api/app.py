"""Demo API. Case reads return inputs. A run is the only path to a report."""

from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from .catalog import find_case, load_catalog
from .service import (
    case_inputs,
    case_list,
    error_document,
    parse_body,
    run_catalog,
    run_paths,
    run_texts,
    submission,
)

_ROOT = Path(__file__).resolve().parents[2]
_FIXTURES = _ROOT / "fixtures"
_CASES = load_catalog(_FIXTURES / "case_catalog.json", _FIXTURES)

app = FastAPI(title="Veriflow demo")


def _missing(case_id: str) -> JSONResponse:
    return JSONResponse(
        error_document(
            "input_error",
            "missing_required_field",
            f"Case {case_id} is not in the catalog.",
        ),
        status_code=404,
    )


@app.get("/api/cases")
def list_cases() -> JSONResponse:
    """Return every catalog case."""

    return JSONResponse(case_list(_CASES))


@app.get("/api/cases/{case_id}")
def read_case(case_id: str) -> JSONResponse:
    """Return one case's graph and transcript text, never a report."""

    case = find_case(_CASES, case_id)
    if case is None:
        return _missing(case_id)
    return JSONResponse(case_inputs(case))


@app.post("/api/cases/{case_id}/run")
def run_case(case_id: str) -> JSONResponse:
    """Evaluate one catalog case from its files."""

    case = find_case(_CASES, case_id)
    if case is None:
        return _missing(case_id)
    graph_path = case.graph_path
    transcript_path = case.transcript_path
    if graph_path is None or transcript_path is None:
        return JSONResponse(
            error_document(
                "input_error",
                "missing_required_field",
                f"Case {case_id} needs both a graph file and a transcript file.",
            )
        )
    return JSONResponse(run_paths(graph_path, transcript_path))


@app.post("/api/run")
async def run_submission(request: Request) -> JSONResponse:
    """Evaluate the graph and transcript text on the page."""

    try:
        graph, transcript = submission(parse_body(await request.body()))
    except ValueError as exc:
        return JSONResponse(
            error_document("input_error", "malformed_json", str(exc)),
            status_code=400,
        )
    return JSONResponse(run_texts(graph, transcript))


@app.post("/api/runs")
def run_cases() -> JSONResponse:
    """Evaluate every case that has both a graph file and a transcript file."""

    return JSONResponse(run_catalog(_CASES))
