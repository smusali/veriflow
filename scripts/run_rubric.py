"""Print the rubric generated for the assignment workflow."""

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv

from src.inputs import load_workflow
from src.judge import build_rubric, openai_completer

load_dotenv(ROOT / ".env")

workflow = load_workflow(ROOT / "fixtures/assignment_graph.json")
rubric = build_rubric(
    workflow,
    openai_completer(os.environ["OPENAI_API_KEY"], os.environ["OPENAI_MODEL"]),
)

for node_id, criteria in rubric.by_node.items():
    print(f"node {node_id}")
    for criterion in criteria:
        print(
            f"  {criterion.id} [{criterion.requirement_type}] "
            f"exact={criterion.exact_wording}"
        )
        print(f"    source: {criterion.source_text}")
        for sub in criterion.subrequirements:
            print(
                f"    {sub.id} [{sub.requirement_type}] "
                f"condition={sub.condition}"
            )
