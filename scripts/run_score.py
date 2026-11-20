"""Print the adherence score for the assignment transcript."""

import os
import sys
from decimal import Decimal
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv

from src.inputs import load_transcript, load_workflow
from src.judge import build_rubric, extract_evidence, openai_completer
from src.verifier import classify_nodes, score_adherence, verify_traversal

load_dotenv(ROOT / ".env")

workflow = load_workflow(ROOT / "fixtures/assignment_graph.json")
transcript = load_transcript(ROOT / "fixtures/assignment_transcript.json")
completer = openai_completer(os.environ["OPENAI_API_KEY"], os.environ["OPENAI_MODEL"])
rubric = build_rubric(workflow, completer)
evidence = extract_evidence(transcript, rubric, completer)
evaluation = classify_nodes(workflow, rubric, evidence, transcript)
traversal = verify_traversal(workflow, evaluation, evidence)
adherence = score_adherence(evaluation, traversal)


def show(score: Decimal | None) -> str:
    if score is None:
        return "null"
    return format(score, "f")


for node in adherence.nodes:
    print(f"{node.node_id} score={show(node.score)}")
print(f"node average: {show(adherence.node_average)}")
print(f"transition score: {show(adherence.transition_score)}")
print(f"adherence: {show(adherence.adherence_score)}")
print(f"verdict: {adherence.verdict}")
