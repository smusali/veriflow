"""Print the evaluation result for the assignment transcript."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from main import evaluate
from src.report import render_report

print(
    render_report(
        evaluate(
            ROOT / "fixtures/assignment_graph.json",
            ROOT / "fixtures/assignment_transcript.json",
        )
    )
)
