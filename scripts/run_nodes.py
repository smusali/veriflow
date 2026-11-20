"""Print one node status for the assignment transcript."""

from common import GRAPH, TRANSCRIPT

from src.evidence import extract_evidence
from src.inputs import load_transcript, load_workflow
from src.nodes import classify_nodes
from src.openai import openai_completer
from src.rubric import build_rubric

workflow = load_workflow(GRAPH)
transcript = load_transcript(TRANSCRIPT)
completer = openai_completer()
rubric = build_rubric(workflow, completer)
evidence = extract_evidence(transcript, rubric, completer)
evaluation = classify_nodes(workflow, rubric, evidence, transcript)

for result in evaluation.results:
    if result.evidence_turns:
        turns = ", ".join(str(index) for index in result.evidence_turns)
    else:
        turns = "none"
    print(
        f"{result.node_id} [{result.applicability}/{result.status}] "
        f"attempted={result.attempted} "
        f"credit={result.satisfied_count}/{result.applicable_count} "
        f"turns={turns}"
    )
    if result.missing_requirements:
        print(f"  missing: {', '.join(result.missing_requirements)}")
    if result.violated_requirements:
        print(f"  violated: {', '.join(result.violated_requirements)}")
    print(f"  {result.explanation}")
