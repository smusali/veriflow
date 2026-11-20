"""Print one node status for the assignment transcript."""

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv

from src.inputs import load_transcript, load_workflow
from src.judge import build_rubric, extract_evidence, openai_completer
from src.verifier import classify_nodes

load_dotenv(ROOT / ".env")

workflow = load_workflow(ROOT / "fixtures/assignment_graph.json")
transcript = load_transcript(ROOT / "fixtures/assignment_transcript.json")
completer = openai_completer(os.environ["OPENAI_API_KEY"], os.environ["OPENAI_MODEL"])
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
