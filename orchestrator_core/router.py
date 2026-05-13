"""Deterministic control-routing layer for orchestrator artifacts."""

from __future__ import annotations

import random
from typing import Sequence

from .adapters import DeepNeuralEIEAdapter
from .artifact_bus import ArtifactBus
from .memory_store import StructuralMemoryStore
from .schemas import ArtifactEvaluation, CandidateArtifact, OrchestratorDecision, SystemEdge, SystemNode
from .system_graph import SystemGraph


class DeterministicRouter:
    def __init__(
        self,
        *,
        graph: SystemGraph,
        bus: ArtifactBus,
        eie_adapter: DeepNeuralEIEAdapter,
        memory_store: StructuralMemoryStore,
        seed: int,
        memory_node_id: str = "structural_memory",
        evaluator_node_id: str = "eie_evaluator",
    ) -> None:
        self.graph = graph
        self.bus = bus
        self.eie_adapter = eie_adapter
        self.memory_store = memory_store
        self.seed = int(seed)
        self._rng = random.Random(seed)
        self.memory_node_id = memory_node_id
        self.evaluator_node_id = evaluator_node_id

    def route_artifact(
        self,
        artifact: CandidateArtifact,
        evaluations: Sequence[ArtifactEvaluation],
        *,
        require_hidden_validation: bool = True,
    ) -> OrchestratorDecision:
        self._ensure_producer_node(artifact)
        self._ensure_artifact_node(artifact)
        self._ensure_evaluator_node()
        self.bus.submit(artifact, acceptance_requested=not artifact.uses_test_signal)
        self._add_edge_once(
            SystemEdge(
                edge_id=f"produces:{artifact.producer}:{artifact.artifact_id}",
                source_node_id=artifact.producer,
                target_node_id=f"artifact:{artifact.artifact_id}",
                edge_type="produces",
                artifact_type=artifact.artifact_type,
                metadata={"generation": artifact.created_generation},
            )
        )
        for index, evaluation in enumerate(evaluations):
            acceptance_driving = (
                not artifact.uses_test_signal
                and not evaluation.uses_test_signal
                and evaluation.split in {"validation", "hidden_validation"}
            )
            self.bus.record_evaluation(evaluation, acceptance_driving=acceptance_driving)
            self._add_edge_once(
                SystemEdge(
                    edge_id=f"evaluates:{artifact.artifact_id}:{evaluation.split}:{index}",
                    source_node_id=f"artifact:{artifact.artifact_id}",
                    target_node_id=self.evaluator_node_id,
                    edge_type="routes_to",
                    artifact_type=artifact.artifact_type,
                    metadata={
                        "split": evaluation.split,
                        "passed": evaluation.passed,
                        "acceptance_driving": acceptance_driving,
                    },
                )
            )

        residue_offset = len(self.eie_adapter.routed_residues)
        mutation_offset = len(self.eie_adapter.mutation_decisions)
        decision = self.eie_adapter.process_artifact(
            artifact,
            evaluations,
            generation=artifact.created_generation,
            require_hidden_validation=require_hidden_validation,
        )

        for residue in self.eie_adapter.routed_residues[residue_offset:]:
            residue_node_id = f"failure_residue:{residue.residue_id}"
            self.graph.add_node(
                SystemNode(
                    node_id=residue_node_id,
                    node_type="failure_residue",
                    repo_name=None,
                    module_name="eie_core.FailureResidue",
                    capabilities=("failure_residue",),
                    metadata=residue.to_dict(),
                )
            )
            self._add_edge_once(
                SystemEdge(
                    edge_id=f"residue:{artifact.artifact_id}:{residue.residue_id}",
                    source_node_id=f"artifact:{artifact.artifact_id}",
                    target_node_id=residue_node_id,
                    edge_type="produces",
                    artifact_type=artifact.artifact_type,
                    metadata={"split": residue.split, "failure_type": residue.failure_type},
                )
            )
            self._add_edge_once(
                SystemEdge(
                    edge_id=f"routes_to_eie:{residue.residue_id}",
                    source_node_id=residue_node_id,
                    target_node_id=self.evaluator_node_id,
                    edge_type="routes_to",
                    artifact_type=artifact.artifact_type,
                    metadata={"acceptance_driving": residue.split != "test"},
                )
            )
            self.memory_store.store_failure_residue(residue)

        for mutation_decision in self.eie_adapter.mutation_decisions[mutation_offset:]:
            self.memory_store.store_mutation_decision(mutation_decision)

        if decision.accepted:
            version_node_id = f"problem_space_version:{decision.problem_space_version_id}"
            if decision.problem_space_version_id and not self.graph.has_node(version_node_id):
                self.graph.add_node(
                    SystemNode(
                        node_id=version_node_id,
                        node_type="problem_space_version",
                        repo_name=None,
                        module_name="eie_core.ProblemSpaceVersion",
                        capabilities=("problem_space_lineage",),
                        metadata={"version_id": decision.problem_space_version_id},
                    )
                )
            if decision.problem_space_version_id:
                self._add_edge_once(
                    SystemEdge(
                        edge_id=f"accepts:{artifact.artifact_id}:{decision.problem_space_version_id}",
                        source_node_id=f"artifact:{artifact.artifact_id}",
                        target_node_id=version_node_id,
                        edge_type="accepts",
                        artifact_type=artifact.artifact_type,
                        metadata={"decision_id": decision.decision_id},
                    )
                )
        else:
            self._add_edge_once(
                SystemEdge(
                    edge_id=f"rejects:{artifact.artifact_id}:{decision.decision_id}",
                    source_node_id=f"artifact:{artifact.artifact_id}",
                    target_node_id=self.memory_node_id,
                    edge_type="rejects",
                    artifact_type=artifact.artifact_type,
                    metadata={"rejection_reason": decision.rejection_reason},
                )
            )

        self._add_edge_once(
            SystemEdge(
                edge_id=f"stores:{artifact.artifact_id}:{decision.decision_id}",
                source_node_id=self.memory_node_id,
                target_node_id=f"artifact:{artifact.artifact_id}",
                edge_type="stores",
                artifact_type=artifact.artifact_type,
                metadata={"accepted": decision.accepted},
            )
        )
        lineage = self.graph.artifact_lineage(artifact.artifact_id)
        stored_decision = self.memory_store.store_decision(artifact, decision, lineage)
        self.bus.record_decision(stored_decision)
        return stored_decision

    def _ensure_producer_node(self, artifact: CandidateArtifact) -> None:
        if self.graph.has_node(artifact.producer):
            return
        self.graph.add_node(
            SystemNode(
                node_id=artifact.producer,
                node_type="candidate_generator",
                repo_name=artifact.source_repo,
                module_name="orchestrator_core.adapters.HybridProgramSynthesisAdapter",
                capabilities=artifact.declared_capabilities,
                metadata={"seed": self.seed},
            )
        )

    def _ensure_artifact_node(self, artifact: CandidateArtifact) -> None:
        node_id = f"artifact:{artifact.artifact_id}"
        if self.graph.has_node(node_id):
            return
        self.graph.add_node(
            SystemNode(
                node_id=node_id,
                node_type="artifact",
                repo_name=artifact.source_repo,
                module_name="orchestrator_core.schemas.CandidateArtifact",
                capabilities=artifact.declared_capabilities,
                metadata=artifact.to_dict(),
            )
        )

    def _ensure_evaluator_node(self) -> None:
        if self.graph.has_node(self.evaluator_node_id):
            return
        self.graph.add_node(
            SystemNode(
                node_id=self.evaluator_node_id,
                node_type="evaluator",
                repo_name="DeepNeural-AutoExploration",
                module_name="orchestrator_core.adapters.DeepNeuralEIEAdapter",
                capabilities=("failure_residue_routing", "validation_gate"),
                metadata={"seed": self.seed},
            )
        )

    def _add_edge_once(self, edge: SystemEdge) -> None:
        if edge.edge_id in self.graph.edges:
            return
        self.graph.add_edge(edge)
