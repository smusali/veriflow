"""Evaluate a transcript against a workflow graph."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

from src.inputs import ContractError, load_transcript, load_workflow
from src.judge import JudgeError, build_rubric, extract_evidence, openai_completer
from src.report import EvaluationReport, assemble_report, render_report
from src.verifier import classify_nodes, score_adherence, verify_traversal

_EVALUATION_ERROR = "evaluation_error"


@dataclass(frozen=True, slots=True)
class _Arguments:
    graph: Path
    transcript: Path
    output: Path


def evaluate(graph: Path, transcript: Path) -> EvaluationReport:
    """Validate the inputs, then judge, classify, traverse, and score them."""

    workflow = load_workflow(graph)
    conversation = load_transcript(transcript)
    completer = openai_completer()
    rubric = build_rubric(workflow, completer)
    evidence = extract_evidence(conversation, rubric, completer)
    evaluation = classify_nodes(workflow, rubric, evidence, conversation)
    traversal = verify_traversal(workflow, evaluation, evidence)
    adherence = score_adherence(evaluation, traversal)
    return assemble_report(workflow, evaluation, traversal, adherence, evidence)


def _existing_file(value: str) -> Path:
    path = Path(value)
    if not path.is_file():
        raise argparse.ArgumentTypeError(f"File not found: {path}")
    return path


def _path_argument(value: object, flag: str) -> Path:
    if isinstance(value, Path):
        return value
    raise TypeError(f"{flag} must be a path.")


def _parse_args(argv: list[str] | None) -> _Arguments:
    parser = argparse.ArgumentParser(
        description="Evaluate a transcript against a workflow graph.",
    )
    parser.add_argument(
        "--graph",
        type=_existing_file,
        required=True,
        help="Path to the workflow graph JSON.",
    )
    parser.add_argument(
        "--transcript",
        type=_existing_file,
        required=True,
        help="Path to the transcript JSON.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Path to write the result JSON.",
    )
    parsed = parser.parse_args(argv)
    return _Arguments(
        graph=_path_argument(parsed.graph, "--graph"),
        transcript=_path_argument(parsed.transcript, "--transcript"),
        output=_path_argument(parsed.output, "--output"),
    )


def _write_error(path: Path, outcome: str, code: str, message: str) -> None:
    print(message, file=sys.stderr)
    document: dict[str, dict[str, str]] = {
        "error": {
            "outcome": outcome,
            "code": code,
            "message": message,
        }
    }
    path.write_text(
        json.dumps(document, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def main(argv: list[str] | None = None) -> int:
    """Parse paths, run the evaluation, and write the result JSON."""

    args = _parse_args(argv)
    if not args.output.parent.is_dir():
        print(
            f"Output directory does not exist: {args.output.parent}",
            file=sys.stderr,
        )
        return 2
    try:
        report = evaluate(args.graph, args.transcript)
    except ContractError as exc:
        _write_error(args.output, exc.outcome.value, exc.code.value, str(exc))
        return 1
    except JudgeError as exc:
        _write_error(args.output, _EVALUATION_ERROR, exc.code.value, str(exc))
        return 1
    args.output.write_text(render_report(report), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
