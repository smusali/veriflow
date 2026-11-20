"""Run the evaluation pipeline from a graph file and a transcript file."""

from __future__ import annotations

from pathlib import Path

from .evidence import extract_evidence
from .inputs import load_transcript, load_workflow
from .nodes import classify_nodes
from .openai import openai_completer
from .report import EvaluationReport, assemble_report
from .rubric import build_rubric
from .score import score_adherence
from .traversal import verify_traversal


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


__all__ = ["evaluate"]
