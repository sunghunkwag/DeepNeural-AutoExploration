import copy
import json
from pathlib import Path

import pytest
import torch

from benchmarks.code_level_rsi_benchmark import (
    _collect_program_transitions,
    _mode_config,
    _validate_candidate_program,
    run as run_code_level_rsi,
)
from benchmarks.recursive_self_improvement_benchmark import (
    WrongWorldModel,
    _build_memory,
    _initial_state,
    _make_model,
    _train_world_model_from_operator_transitions,
)
from candidate_sandbox import CandidateSandbox
from cognitive_core import EpisodicMemory, MemoryRecord, WorldModel
from learned_task_encoder import LearnedTaskEncoder, TaskConditionedRegressor, train_task_encoder
from operator_dsl import (
    OperatorProgram,
    PrimitiveStep,
    ProgramGenome,
    assert_no_task_family_branching,
    assert_program_diversity,
    compile_operator_program,
    default_operator_programs,
    execute_operator_program,
)
from scripts.verify_code_level_rsi_smoke import verify as verify_code_level_smoke
from task_suite import ProceduralTaskSuite


def _fixture(seed=501, tmp_path=None):
    cfg = _mode_config("smoke")
    state = _initial_state(seed)
    suite = ProceduralTaskSuite(seed=seed)
    hidden = ProceduralTaskSuite(seed=seed + 10000)
    train = suite.sample_mixed_tasks("train", int(cfg["train"]))
    val = suite.sample_mixed_tasks("validation", int(cfg["validation"]))
    hidden_val = hidden.sample_mixed_tasks("validation", int(cfg["hidden_validation"]))
    test = suite.sample_mixed_tasks("test", int(cfg["test"]), ood=True)
    encoder = LearnedTaskEncoder(latent_dim=8, hidden_dim=32, seed=seed)
    decoder = TaskConditionedRegressor(latent_dim=8, hidden_dim=32)
    train_task_encoder(encoder, decoder, train, val, steps=2, lr=0.01)
    memory = _build_memory(train, state, encoder)
    genome = ProgramGenome.default()
    transitions, _, summary = _collect_program_transitions(train, state, encoder, memory, genome)
    world_model, _, _ = _train_world_model_from_operator_transitions(seed + 11, transitions)
    sandbox = CandidateSandbox(tmp_path or Path.cwd() / ".candidate_sandbox", timeout_seconds=3.0, max_loss=1e5)
    return state, train, val, hidden_val, test, encoder, memory, genome, world_model, sandbox, summary


def test_candidate_programs_compile_execute_and_change_behavior(tmp_path):
    state, _, val, _, _, encoder, memory, _, world_model, _, _ = _fixture(tmp_path=tmp_path)
    base = default_operator_programs()[0]
    candidate = OperatorProgram(
        "candidate_clip_decay",
        (
            PrimitiveStep("gradient_norm_clipping", {"max_norm": 2.0}),
            PrimitiveStep("constant_lr"),
            PrimitiveStep("maml_step"),
            PrimitiveStep("decayed_lr", {"decay": 0.75}),
            PrimitiveStep("variable_lr_step", {"decay": 0.75}),
        ),
        {"lr_scale": 0.9},
        base.program_id,
        1,
    )
    compile_operator_program(candidate)
    base_trace = execute_operator_program(
        base,
        __import__("benchmarks.code_level_rsi_benchmark", fromlist=["_make_contexts"])._make_contexts([val[0]], state, encoder, memory, world_model)[0],
    )
    candidate_trace = execute_operator_program(
        candidate,
        __import__("benchmarks.code_level_rsi_benchmark", fromlist=["_make_contexts"])._make_contexts([val[0]], state, encoder, memory, world_model)[0],
    )
    assert base_trace.query_loss_after != candidate_trace.query_loss_after
    assert candidate_trace.params["primitive_sequence"] != base_trace.params["primitive_sequence"]


def test_candidate_adaptation_does_not_access_query_y(tmp_path):
    class QueryPoisonTask:
        task_id = "poison"
        family = "poison"
        split = "validation"
        support_x = torch.linspace(-1, 1, 5).reshape(-1, 1)
        support_y = torch.sin(support_x)

        @property
        def query_x(self):
            raise AssertionError("query_x should not be needed during adaptation")

        @property
        def query_y(self):
            raise AssertionError("query_y leaked into adaptation")

    from adaptation_operators import OperatorExecutionContext

    state = _initial_state(502)
    task = QueryPoisonTask()
    encoder = LearnedTaskEncoder(latent_dim=8, seed=502)
    ctx = OperatorExecutionContext(
        model=_make_model(state),
        task=task,
        task_embedding=encoder.infer(task.support_x, task.support_y),
        base_inner_lr=float(state["inner_lr"]),
        inner_steps=1,
    )
    program = OperatorProgram("support_only", (PrimitiveStep("constant_lr"), PrimitiveStep("maml_step")), {"lr_scale": 1.0})
    adapted, details = compile_operator_program(program).adapt_params(ctx)
    assert adapted
    assert details.selected_lrs


def test_candidate_acceptance_rejects_empty_or_test_validation(tmp_path):
    state, _, val, hidden_val, test, encoder, memory, genome, world_model, sandbox, _ = _fixture(tmp_path=tmp_path)
    candidate = OperatorProgram("candidate", (PrimitiveStep("constant_lr"), PrimitiveStep("maml_step")), {"lr_scale": 0.8}, genome.active_program_id, 1)
    with pytest.raises(ValueError):
        _validate_candidate_program(genome, candidate, [], hidden_val, state, encoder, memory, world_model, sandbox, generation=1, seed=1)
    with pytest.raises(ValueError):
        _validate_candidate_program(genome, candidate, test, hidden_val, state, encoder, memory, world_model, sandbox, generation=1, seed=1)


def test_rejected_candidate_rolls_back_exactly(tmp_path):
    state, _, val, hidden_val, _, encoder, memory, genome, world_model, sandbox, _ = _fixture(tmp_path=tmp_path)
    before = copy.deepcopy(genome.snapshot())
    bad = OperatorProgram("nan_candidate", (PrimitiveStep("constant_lr"), PrimitiveStep("maml_step")), {"lr_scale": float("nan")}, genome.active_program_id, 1)
    decision = _validate_candidate_program(genome, bad, val, hidden_val, state, encoder, memory, world_model, sandbox, generation=1, seed=1)
    assert not decision["accepted"]
    assert decision["rollback"]
    assert genome.snapshot() == before


def test_shuffled_memory_damages_memory_gated_program(tmp_path):
    state, _, val, _, _, encoder, _, _, _, _, _ = _fixture(tmp_path=tmp_path)
    task = val[0]
    z = encoder.infer(task.support_x, task.support_y)
    good = EpisodicMemory(capacity=4)
    bad = EpisodicMemory(capacity=4)
    good.add(MemoryRecord("good", torch.zeros(1), torch.zeros(1), 0.9, 0.0, z, split="train"))
    bad.add(MemoryRecord("bad", torch.zeros(1), torch.zeros(1), -0.9, 1.0, z.roll(1), split="train"))
    program = [p for p in default_operator_programs() if p.program_id == "dsl0_memory_gate"][0]
    from benchmarks.code_level_rsi_benchmark import _make_contexts

    good_trace = execute_operator_program(program, _make_contexts([task], state, encoder, good, None)[0])
    bad_trace = execute_operator_program(program, _make_contexts([task], state, encoder, bad, None)[0])
    assert good_trace.memory_reward_signal > bad_trace.memory_reward_signal
    assert good_trace.selected_lrs != bad_trace.selected_lrs


def test_wrong_world_model_changes_world_model_gated_program(tmp_path):
    class PreferHighLR(WorldModel):
        def forward(self, latent_state, action):
            out = torch.zeros_like(latent_state)
            out[..., 0] = action[..., 0]
            return out

    state, _, val, _, _, encoder, memory, _, _, _, _ = _fixture(tmp_path=tmp_path)
    program = [p for p in default_operator_programs() if p.program_id == "dsl0_world_lr"][0]
    from benchmarks.code_level_rsi_benchmark import _make_contexts

    good_trace = execute_operator_program(program, _make_contexts([val[0]], state, encoder, memory, PreferHighLR(8, 6))[0])
    wrong_trace = execute_operator_program(program, _make_contexts([val[0]], state, encoder, memory, WrongWorldModel(8, 6))[0])
    assert good_trace.selected_lrs != wrong_trace.selected_lrs


def test_dead_code_and_task_family_hardcoding_detectors():
    with pytest.raises(RuntimeError):
        assert_program_diversity({"a": [1.0, 1.0], "b": [1.0, 1.0]})
    source = Path("operator_dsl.py").read_text(encoding="utf-8") + Path("rsi_candidate_generator.py").read_text(encoding="utf-8") + Path("benchmarks/code_level_rsi_benchmark.py").read_text(encoding="utf-8")
    assert_no_task_family_branching(source)


def test_nan_and_exploding_candidates_are_rejected(tmp_path):
    state, _, val, _, _, encoder, memory, _, world_model, _, _ = _fixture(tmp_path=tmp_path)
    from benchmarks.code_level_rsi_benchmark import _make_contexts

    contexts = _make_contexts(val, state, encoder, memory, world_model)
    sandbox = CandidateSandbox(tmp_path, timeout_seconds=3.0, max_loss=1e-9)
    nan_program = OperatorProgram("nan", (PrimitiveStep("constant_lr"), PrimitiveStep("maml_step")), {"lr_scale": float("nan")})
    exploding_program = OperatorProgram("explode", (PrimitiveStep("constant_lr"), PrimitiveStep("maml_step")), {"lr_scale": 1.0})
    assert not sandbox.evaluate(nan_program, contexts).ok
    exploding = sandbox.evaluate(exploding_program, contexts)
    assert not exploding.ok
    assert exploding.rejected_reason == "exploding_loss"


def test_code_level_smoke_manifest_verifier_and_reuse(tmp_path):
    out = tmp_path / "code_level_rsi_smoke_seed42.json"
    result = run_code_level_rsi("smoke", 42, str(out))
    verify_code_level_smoke(out)
    assert out.exists()
    manifest = json.loads(Path(result["manifest_path"]).read_text(encoding="utf-8"))
    aggregate = result["aggregate"]
    assert aggregate["candidate_count"]["mean"] > 0
    assert aggregate["compiled_candidate_count"]["mean"] > 0
    assert aggregate["accepted_program_count"]["mean"] > 0
    assert aggregate["accepted_program_reuse_count"]["mean"] > 0
    assert manifest["accepted_candidates"]
    assert manifest["rejected_candidates"]
    assert all("rejection_reason" in item for item in manifest["accepted_candidates"] + manifest["rejected_candidates"])


def test_code_level_smoke_is_deterministic_for_same_seed(tmp_path):
    first = run_code_level_rsi("smoke", 77, str(tmp_path / "first.json"))
    second = run_code_level_rsi("smoke", 77, str(tmp_path / "second.json"))
    for key in ("candidate_count", "compiled_candidate_count", "accepted_program_count", "rollback_count", "runtime_behavior_difference_observed"):
        assert first["aggregate"][key]["mean"] == pytest.approx(second["aggregate"][key]["mean"], abs=1e-8)
    first_ids = [item["candidate"]["program_id"] for item in first["generated_candidates"]]
    second_ids = [item["candidate"]["program_id"] for item in second["generated_candidates"]]
    assert first_ids == second_ids


def test_readme_code_level_rsi_claims_are_honest():
    readme = Path("README.md").read_text(encoding="utf-8")
    assert "not AGI" in readme
    assert "not human-level intelligence" in readme
    assert "not a technological singularity" in readme or "not a singularity system" in readme
    assert "code-level" in readme
    assert "open-ended autonomous recursive self-improvement" in readme
    assert "generate bounded operator programs" in readme

