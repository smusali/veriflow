"""Print the workflow path for the assignment transcript."""

from common import GRAPH, TRANSCRIPT

from src.evidence import extract_evidence
from src.inputs import load_transcript, load_workflow
from src.nodes import classify_nodes
from src.openai import openai_completer
from src.rubric import build_rubric
from src.traversal import verify_traversal

workflow = load_workflow(GRAPH)
transcript = load_transcript(TRANSCRIPT)
completer = openai_completer()
rubric = build_rubric(workflow, completer)
evidence = extract_evidence(transcript, rubric, completer)
evaluation = classify_nodes(workflow, rubric, evidence, transcript)
traversal = verify_traversal(workflow, evaluation, evidence)


def show(path: tuple[str, ...] | None) -> str:
    if path is None:
        return "unresolved"
    return "[" + ", ".join(path) + "]"


print("candidates: " + "; ".join(show(path) for path in traversal.candidate_paths))
print(f"expected: {show(traversal.expected_path)}")
print(f"observed: {show(traversal.observed_path)}")
print(f"validity: {traversal.validity}")
if traversal.omitted_nodes:
    print("omitted: " + ", ".join(traversal.omitted_nodes))
if traversal.unreached_nodes:
    print("unreached: " + ", ".join(traversal.unreached_nodes))
if traversal.transitions:
    for transition in traversal.transitions:
        print(f"  {transition.source} -> {transition.target} [{transition.assessment}]")
else:
    print("transitions: none")
for event in evidence.stage_events:
    print(f"event node {event.node_id} turn {event.turn} start {event.start}")
