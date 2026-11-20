"""Print criterion findings for the assignment transcript."""

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv

from src.inputs import load_transcript, load_workflow
from src.judge import build_rubric, extract_evidence, openai_completer

load_dotenv(ROOT / ".env")

workflow = load_workflow(ROOT / "fixtures/assignment_graph.json")
transcript = load_transcript(ROOT / "fixtures/assignment_transcript.json")
completer = openai_completer(os.environ["OPENAI_API_KEY"], os.environ["OPENAI_MODEL"])
rubric = build_rubric(workflow, completer)
evidence = extract_evidence(transcript, rubric, completer)

for finding in evidence.findings:
    if finding.evidence_turns:
        turns = ", ".join(str(index) for index in finding.evidence_turns)
    else:
        turns = "none"
    if finding.action_turn is None:
        action = "none"
    else:
        action = str(finding.action_turn)
    print(
        f"{finding.criterion_id} [{finding.applicability}/{finding.result}] "
        f"turns={turns} action={action}"
    )
    print(f"  {finding.reason}")
