from failure_residue import MissingOperatorInferencer, ResidueExtractor
from neural_search.representation_probe import RepresentationProbeResult, representation_failures_to_traces


def test_probe_failures_produce_actionable_failure_residues():
    probe = RepresentationProbeResult(
        genome_id="arch_probe",
        activation_variance_by_module={"input_layer": 0.0},
        representation_collapse_score=1.0,
        task_embedding_separability=0.0,
        hidden_state_similarity=0.99,
        module_ablation_contribution={"input_layer": 0.0},
        failure_modes=["representation collapse", "task-family entanglement"],
        recommended_mutations={
            "representation collapse": ["wider_residual_stack"],
            "task-family entanglement": ["object_binding_adapter"],
        },
    )
    traces = representation_failures_to_traces(probe, benchmark_name="probe_test", seed=701)
    residues = ResidueExtractor().extract_many(traces)
    assert {residue.failure_type for residue in residues} == {"representation collapse", "task-family entanglement"}
    assert all(residue.used_heldout_labels is False for residue in residues)
    hypotheses = MissingOperatorInferencer(min_support=1).infer(residues)
    families = {hypothesis.proposed_module_family for hypothesis in hypotheses}
    assert {"wider_residual_stack", "object_binding_adapter"} <= families

