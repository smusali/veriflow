"""Print the evaluation result for the assignment transcript."""

from common import GRAPH, TRANSCRIPT

from src.evaluate import evaluate
from src.report import render_report

print(render_report(evaluate(GRAPH, TRANSCRIPT)))
