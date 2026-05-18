from world_model_v2 import (
    CausalGraphState,
    CounterfactualRollout,
    Intervention,
    InterventionModel,
    ObjectRecord,
    ObjectRelation,
    ObjectState,
    UncertaintyDecomposition,
)


def test_object_state_creation_and_relations():
    state = ObjectState.from_objects(
        [
            ObjectRecord("agent", {"x": 0.0, "vx": 1.0}),
            ObjectRecord("target", {"x": 5.0}),
        ],
        [ObjectRelation("agent", "target", "sees")],
    )
    assert state.get_attribute("agent", "x") == 0.0
    assert state.relations[0].relation_type == "sees"


def test_intervention_application_is_do_style_and_non_mutating():
    state = ObjectState().add_object("agent", {"x": 0.0, "vx": 1.0})
    changed = InterventionModel().apply(state, [Intervention("agent", "vx", 3.0)])
    assert state.get_attribute("agent", "vx") == 1.0
    assert changed.get_attribute("agent", "vx") == 3.0


def test_counterfactual_rollout_changes_object_state():
    state = ObjectState().add_object("agent", {"x": 0.0, "vx": 1.0})
    comparison = CounterfactualRollout().compare(state, [Intervention("agent", "vx", 3.0)], horizon=2)
    assert comparison.observed[-1].get_attribute("agent", "x") == 2.0
    assert comparison.counterfactual[-1].get_attribute("agent", "x") == 6.0
    assert comparison.attribute_deltas["agent.x"] == 4.0


def test_causal_graph_and_uncertainty_components():
    graph = CausalGraphState().add_edge("agent.x", "target.x", weight=0.5, latent_confounder=True)
    assert graph.parents("target.x") == ["agent.x"]
    assert graph.has_latent_confounder("agent.x", "target.x")
    components = UncertaintyDecomposition().decompose(
        model_scores=[0.1, 0.3],
        observation_errors=[0.2, 0.2],
        intervention_effects=[1.0, 3.0],
    )
    assert components.model_uncertainty > 0.0
    assert components.observation_uncertainty == 0.0
    assert components.intervention_uncertainty > 0.0
