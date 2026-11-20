"""Print criterion findings for the assignment transcript."""

from common import GRAPH, TRANSCRIPT

from src.evidence import extract_evidence
from src.inputs import load_transcript, load_workflow
from src.openai import openai_completer
from src.rubric import build_rubric

workflow = load_workflow(GRAPH)
transcript = load_transcript(TRANSCRIPT)
completer = openai_completer()
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
