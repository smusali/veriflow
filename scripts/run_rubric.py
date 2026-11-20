"""Print the rubric generated for the assignment workflow."""

from common import GRAPH

from src.inputs import load_workflow
from src.openai import openai_completer
from src.rubric import build_rubric

workflow = load_workflow(GRAPH)
rubric = build_rubric(workflow, openai_completer())

for node_id, criteria in rubric.by_node.items():
    print(f"node {node_id}")
    for criterion in criteria:
        print(f"  {criterion.id} [{criterion.requirement_type}] exact={criterion.exact_wording}")
        print(f"    source: {criterion.source_text}")
        for sub in criterion.subrequirements:
            print(f"    {sub.id} [{sub.requirement_type}] condition={sub.condition}")
