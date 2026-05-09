"""Sandboxed candidate evaluation for bounded operator programs."""

from __future__ import annotations

import concurrent.futures
import json
import math
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from statistics import mean
from typing import Dict, List, Optional, Sequence

from adaptation_operators import OperatorExecutionContext
from operator_dsl import OperatorProgram, compile_operator_program


@dataclass
class CandidateSandboxResult:
    program_id: str
    compiled: bool
    ok: bool
    scores: List[float] = field(default_factory=list)
    improvements: List[float] = field(default_factory=list)
    traces: List[Dict[str, object]] = field(default_factory=list)
    behavior_signature: List[float] = field(default_factory=list)
    error: Optional[str] = None
    rejected_reason: Optional[str] = None
    elapsed_seconds: float = 0.0
    sandbox_dir: str = ""

    def to_dict(self) -> Dict[str, object]:
        return {
            "program_id": self.program_id,
            "compiled": self.compiled,
            "ok": self.ok,
            "scores": [float(x) for x in self.scores],
            "improvements": [float(x) for x in self.improvements],
            "traces": list(self.traces),
            "behavior_signature": [float(x) for x in self.behavior_signature],
            "error": self.error,
            "rejected_reason": self.rejected_reason,
            "elapsed_seconds": float(self.elapsed_seconds),
            "sandbox_dir": self.sandbox_dir,
        }


class CandidateSandbox:
    """Evaluate candidate programs without mutating the main genome.

    The sandbox uses an isolated temporary directory under the repository and
    object-level isolation.  It does not execute shell commands or generated
    source code.  Timeouts, exceptions, NaN/inf losses, and exploding losses are
    surfaced as rejected candidate results.
    """

    def __init__(self, root: str | Path | None = None, timeout_seconds: float = 5.0, max_loss: float = 1e5):
        self.root = Path(root or Path.cwd() / ".candidate_sandbox")
        self.root.mkdir(parents=True, exist_ok=True)
        self.timeout_seconds = float(timeout_seconds)
        self.max_loss = float(max_loss)

    def compile(self, program: OperatorProgram) -> CandidateSandboxResult:
        start = time.monotonic()
        with tempfile.TemporaryDirectory(prefix="candidate_", dir=self.root) as tmp:
            try:
                compiled = compile_operator_program(program)
                payload = {
                    "program": program.to_dict(),
                    "compiled_operator": compiled.name,
                    "source_hash": program.source_hash,
                }
                Path(tmp, "candidate_program.json").write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
                return CandidateSandboxResult(program.program_id, True, True, elapsed_seconds=time.monotonic() - start, sandbox_dir=tmp)
            except Exception as exc:
                return CandidateSandboxResult(program.program_id, False, False, error=repr(exc), rejected_reason="compile_failed", elapsed_seconds=time.monotonic() - start, sandbox_dir=tmp)

    def evaluate(self, program: OperatorProgram, contexts: Sequence[OperatorExecutionContext]) -> CandidateSandboxResult:
        start = time.monotonic()
        with tempfile.TemporaryDirectory(prefix="candidate_", dir=self.root) as tmp:
            try:
                compile_operator_program(program)
            except Exception as exc:
                return CandidateSandboxResult(program.program_id, False, False, error=repr(exc), rejected_reason="compile_failed", elapsed_seconds=time.monotonic() - start, sandbox_dir=tmp)

            def _run() -> CandidateSandboxResult:
                operator = compile_operator_program(program)
                scores: List[float] = []
                improvements: List[float] = []
                traces: List[Dict[str, object]] = []
                signature_accum: List[List[float]] = []
                for context in contexts:
                    from adaptation_operators import OperatorGene

                    gene = OperatorGene(program.program_id, operator.name, {"program": program.to_dict()}, program.generation_created, program.parent_program_id, program.accepted)
                    trace = operator.apply(gene, context)
                    if not math.isfinite(trace.query_loss_after):
                        return CandidateSandboxResult(program.program_id, True, False, rejected_reason="non_finite_loss", sandbox_dir=tmp)
                    if trace.query_loss_after > self.max_loss:
                        return CandidateSandboxResult(program.program_id, True, False, rejected_reason="exploding_loss", sandbox_dir=tmp)
                    scores.append(float(trace.query_loss_after))
                    improvements.append(float(trace.improvement))
                    trace_dict = trace.to_dict()
                    traces.append(trace_dict)
                    behavior = trace_dict.get("params", {}).get("behavior_signature", []) if isinstance(trace_dict.get("params"), dict) else []
                    if behavior:
                        signature_accum.append([float(x) for x in behavior])
                behavior_signature = []
                if signature_accum:
                    cols = len(signature_accum[0])
                    behavior_signature = [float(mean(row[i] for row in signature_accum)) for i in range(cols)]
                return CandidateSandboxResult(program.program_id, True, True, scores=scores, improvements=improvements, traces=traces, behavior_signature=behavior_signature, sandbox_dir=tmp)

            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(_run)
                try:
                    result = future.result(timeout=self.timeout_seconds)
                except concurrent.futures.TimeoutError:
                    future.cancel()
                    return CandidateSandboxResult(program.program_id, True, False, rejected_reason="timeout", elapsed_seconds=time.monotonic() - start, sandbox_dir=tmp)
                except Exception as exc:
                    return CandidateSandboxResult(program.program_id, True, False, error=repr(exc), rejected_reason="execution_exception", elapsed_seconds=time.monotonic() - start, sandbox_dir=tmp)
            result.elapsed_seconds = time.monotonic() - start
            Path(tmp, "candidate_result.json").write_text(json.dumps(result.to_dict(), indent=2, sort_keys=True), encoding="utf-8")
            return result

