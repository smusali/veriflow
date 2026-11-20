"""Print the evaluation result for the assignment transcript."""

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv

from src.inputs import load_transcript, load_workflow
from src.judge import build_rubric, extract_evidence, openai_completer
from src.report import assemble_report, render_report
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
print(render_report(assemble_report(workflow, evaluation, traversal, adherence, evidence)))
