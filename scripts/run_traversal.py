"""Print the workflow path for the assignment transcript."""

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv

from src.inputs import load_transcript, load_workflow
from src.judge import build_rubric, extract_evidence, openai_completer
from src.verifier import classify_nodes, verify_traversal

load_dotenv(ROOT / ".env")

workflow = load_workflow(ROOT / "fixtures/assignment_graph.json")
transcript = load_transcript(ROOT / "fixtures/assignment_transcript.json")
completer = openai_completer(os.environ["OPENAI_API_KEY"], os.environ["OPENAI_MODEL"])
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
