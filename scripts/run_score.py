"""Print the adherence score for the assignment transcript."""

from decimal import Decimal

from common import GRAPH, TRANSCRIPT

from src.evidence import extract_evidence
from src.inputs import load_transcript, load_workflow
from src.nodes import classify_nodes
from src.openai import openai_completer
from src.rubric import build_rubric
from src.score import score_adherence
from src.traversal import verify_traversal

workflow = load_workflow(GRAPH)
transcript = load_transcript(TRANSCRIPT)
completer = openai_completer()
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
