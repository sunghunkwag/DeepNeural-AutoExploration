from neural_search import ArchitectureGenome
from neural_search.multidomain_evaluator import MultiDomainEvaluator
from neural_search.task_families import assert_disjoint_multidomain_splits, build_multidomain_task_families


def test_hard_task_modes_are_harder_than_simple_modes():
    seed = 311
    simple = build_multidomain_task_families(seed=seed, samples_per_split=8, difficulty="simple")
    hard = build_multidomain_task_families(seed=seed, samples_per_split=8, difficulty="hard")
    genome = ArchitectureGenome(input_dim=4, output_dim=2, hidden_dim=8, seed=seed)
    evaluator = MultiDomainEvaluator(seed=seed, train_steps=2, lr=0.01)
    simple_eval = evaluator.train_and_validate(genome, simple)
    hard_eval = evaluator.train_and_validate(genome, hard)
    assert hard_eval.hidden_validation_loss > simple_eval.hidden_validation_loss
    assert {family.difficulty for family in hard} == {"hard"}


def test_task_family_splits_remain_disjoint_in_hard_mode():
    families = build_multidomain_task_families(seed=312, samples_per_split=8, difficulty="hard")
    assert_disjoint_multidomain_splits(families)
    for family in families:
        assert set(family.splits) == {"train", "validation", "hidden_validation", "heldout_test"}
        assert all(batch.difficulty == "hard" for batch in family.splits.values())

