import gzip
import json
from pathlib import Path

from benchmarks import humaneval_template_rsi_benchmark as coding_benchmark


def _write_humaneval_fixture(root: Path) -> Path:
    data_dir = root / "data"
    data_dir.mkdir(parents=True)
    problem = {
        "task_id": "HumanEvalFixture/0",
        "prompt": "\n\ndef strlen(string: str) -> int:\n    \"\"\"Return length.\"\"\"\n",
        "canonical_solution": "    return len(string)\n",
        "test": "def check(candidate):\n    assert candidate('abc') == 3\n    assert candidate('') == 0\n",
        "entry_point": "strlen",
    }
    with gzip.open(data_dir / "HumanEval.jsonl.gz", "wt", encoding="utf-8") as handle:
        handle.write(json.dumps(problem) + "\n")
    return root


def test_humaneval_template_benchmark_reports_baseline_to_evolved(tmp_path):
    root = _write_humaneval_fixture(tmp_path / "human_eval")
    out = tmp_path / "humaneval_template_rsi_smoke.json"
    result = coding_benchmark.run("smoke", 42, str(out), humaneval_dir=root)
    assert out.exists()
    assert Path(result["manifest_path"]).exists()
    assert result["metrics"]["baseline_pass_at_1"] == 0.0
    assert result["metrics"]["evolved_pass_at_1"] == 1.0
    manifest = json.loads(Path(result["manifest_path"]).read_text(encoding="utf-8"))
    assert "canonical solutions ignored and never executed" in manifest["anti_cheat_checks_passed"]
