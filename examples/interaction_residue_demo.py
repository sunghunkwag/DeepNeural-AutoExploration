"""Run the interaction-residue layer demo.

This demo is intentionally small and deterministic. It shows how a stream of
micro-turn events becomes actions, evaluation metrics, residue records, and
RSI-style evaluator decision records.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from interaction_residue_layer import InteractionResidueLayer, build_demo_trace


def main() -> None:
    layer = InteractionResidueLayer()
    actions = layer.run(build_demo_trace())
    report = layer.evaluate()
    decisions = layer.export_evaluator_decisions(generation=0)

    payload = {
        "actions": [action.to_dict() for action in actions],
        "report": {key: value for key, value in report.items() if key != "residue_objects"},
        "evaluator_decisions": decisions,
        "state": layer.to_dict()["state"],
    }
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
