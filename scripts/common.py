"""Shared paths for the assignment inspection scripts."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

GRAPH = ROOT / "fixtures/assignment_graph.json"
TRANSCRIPT = ROOT / "fixtures/assignment_transcript.json"
