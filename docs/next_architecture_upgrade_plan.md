# Next architecture upgrade plan

## Structural weaknesses in the current scaffold

- Task inference is hand-crafted statistics over support data, so it cannot learn reusable context features across procedural families.
- The world model is mostly demonstrated as a side prediction/planning object; it does not centrally rank update actions before adaptation.
- Episodic memory is retrieved and logged, but adaptation can proceed almost identically without the retrieved episodes.
- Self-improvement is limited to small scalar changes and needs a clearer validation-only mutation protocol with complete logs.
- Benchmark outputs do not yet include a manifest that exposes split IDs, seeds, leakage checks, mutation decisions, and config hash.
- Existing benchmarks do not compare enough ablations to prove that learned context, memory, planning, and mutation are active.

## New modules to add

- `learned_task_encoder.py`: CPU-runnable DeepSets context encoder plus quick procedural training/evaluation helpers.
- `model_based_controller.py`: candidate update/action ranking using a learned `WorldModel`, memory summaries, and task latent vectors.
- `operator_mutation.py`: controlled operator mutation candidates and a validation-gated mutation controller with rollback and immutable logs.
- `experiment_manifest.py`: manifest writer and anti-cheat validator for task splits, seeds, memory records, mutation logs, and benchmark result metadata.
- `benchmarks/next_agi_scaffold_benchmark.py`: smoke/quick/full benchmark with ablations and manifest output.
- `examples/next_closed_loop_demo.py`: transparent vertical slice of the upgraded loop.

## Learned task inference strategy

A DeepSets encoder will encode each support pair `[x_i, y_i]`, aggregate by mean and max so the output is robust to support-set permutations, and project the aggregate to `z_task`. Training uses procedural tasks from multiple families and a support-only reconstruction proxy that predicts query targets from query x and z. Evaluation compares held-out/OOD proxy loss against the current hand-crafted baseline. The encoder API will not accept `query_y` for encoding, and tests will perturb `query_y` to prove the latent is unchanged.

## World model influence on update selection

A model-based controller will create candidate adaptation actions such as inner learning-rate scale, inner steps, memory retrieval k, first-order flag, and planner horizon. It will concatenate `z_task`, a memory summary, and each candidate action vector, then call the learned `WorldModel` to predict the next latent/improvement proxy. The controller ranks candidates by predicted improvement, executes the selected candidate on validation/support-query tasks, logs predicted-vs-actual error, and stores transitions for future world-model training.

## Memory-conditioned adaptation

Memory retrieval will affect adaptation by deriving a memory-conditioned inner learning rate and action prior from similar episodes. Relevant memory with positive rewards should increase or stabilize the selected inner_lr, while irrelevant/corrupted memory should not improve transfer. Retrieval remains cosine similarity over task latents, is capacity bounded, rejects records containing query targets, and can exclude test split records during train/validation control.

## Allowed self-improvement operators

Allowed controlled mutations include: inner_lr scale, inner_steps, exploration policy switch, memory retrieval k, learned-vs-handcrafted encoder switch, task encoder usage on/off, world-model planner horizon, first-order vs second-order MAML, residual block width multiplier metadata, uncertainty weighting coefficient, and intrinsic objective weighting coefficient. Every candidate is evaluated only on validation tasks, accepted only if the threshold and consistency requirement are met, otherwise rolled back from a snapshot.

## Cheating paths to block

- Overlapping train/validation/test IDs.
- Query targets used by the task encoder.
- Query targets stored in memory.
- Test split used for mutation acceptance or controller optimization.
- Final test metric used to accept/reject self-improvement.
- Missing or implicit random seeds.
- OOD mode that does not change distribution.
- Benchmark JSON without config, seeds, and task IDs.
- Hardcoded benchmark result constants or fake logs.
- Mutation history deletion/rewrite.

## Tests proving the upgrade is real

- Learned encoder shape, permutation robustness, support-y sensitivity, query-y independence, train-loss reduction, and held-out comparison to the hand-crafted baseline.
- z_task changes controller/adaptation decisions and planner/memory scores; ablating z_task worsens a quick validation proxy.
- Controller calls the learned world model, wrong world model changes action selection, prediction-vs-actual error is logged, and learned world-model training reduces held-out transition error.
- Relevant memory improves adaptation relative to no memory, wrong memory does not, capacity is enforced, similarity beats insertion order, and query/test targets cannot leak.
- Good synthetic mutation is accepted, bad mutation rolls back, validation tasks are required, test split acceptance is rejected, and logs include candidate, scores, flags, and seed.
- Manifest records all task IDs, rejects overlapping splits/missing seeds/test-split mutation/memory query targets, and changes hash when config changes.
- Smoke benchmark confirms the full connected loop beats at least one ablated baseline and records config/task IDs/seeds.
