# Copyright 2026 D-Wave
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Validate saved embeddings against working QPU graphs; never search for new ones."""

import json
import logging
from dataclasses import dataclass
from pathlib import Path

import networkx as nx
import torch
from dwave.cloud import Client
from dwave.embedding import verify_embedding
from dwave.embedding.exceptions import EmbeddingError
from dwave.system import DWaveSampler, FixedEmbeddingComposite

from src.model import TrainedModel

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SavedEmbedding:
    embedding_id: int
    chains: dict[int, list[int]]

    @property
    def priority(self) -> tuple[int, int, int]:
        lengths = [len(chain) for chain in self.chains.values()]
        return max(lengths), sum(lengths), self.embedding_id


def load_banks(directory: Path) -> dict[str, list[SavedEmbedding]]:
    banks = {}
    for path in sorted(directory.glob("*.json")):
        data = json.loads(path.read_text())
        solver = data["solver_name"]
        if solver in banks:
            raise ValueError(f"Duplicate embedding bank for {solver}.")
        banks[solver] = [
            SavedEmbedding(
                record["embedding_id"],
                {int(node): chain for node, chain in record["embedding"].items()},
            )
            for record in data["embeddings"]
        ]
    if not banks:
        raise ValueError(f"No precomputed embedding banks found in {directory}.")
    return banks


def valid_embeddings(
    records: list[SavedEmbedding], source: nx.Graph, target: nx.Graph
) -> list[SavedEmbedding]:
    accepted = []
    for record in sorted(records, key=lambda item: item.priority):
        try:
            verify_embedding(record.chains, source, target)
        except EmbeddingError as exc:
            logger.warning(
                "Skipping invalid embedding %s: %s", record.embedding_id, exc
            )
        else:
            accepted.append(record)
    return accepted


def discover_solvers(
    banks: dict[str, list[SavedEmbedding]], source: nx.Graph
) -> tuple[list[dict], list[str]]:
    options, unavailable = [], []
    with Client.from_config(client="qpu") as client:
        solvers = {solver.name: solver for solver in client.get_solvers()}
        for name, records in banks.items():
            solver = solvers.get(name)
            if solver is None:
                unavailable.append(
                    f"{name}: not available to the configured Leap account."
                )
                continue
            target = nx.Graph()
            target.add_nodes_from(solver.nodes)
            target.add_edges_from(solver.edges)
            accepted = valid_embeddings(records, source, target)
            if accepted:
                options.append(
                    {
                        "label": f"{name} ({len(accepted)} valid embeddings)",
                        "value": name,
                    }
                )
            else:
                unavailable.append(
                    f"{name}: no saved embeddings survive its current working graph."
                )
    return options, unavailable


def sample_qpu(
    name: str,
    count: int,
    model: TrainedModel,
    banks: dict[str, list[SavedEmbedding]],
) -> tuple[torch.Tensor, dict]:
    if name not in banks:
        raise ValueError("Choose a QPU with a precomputed embedding bank.")
    source = nx.Graph()
    source.add_nodes_from(model.nodes)
    source.add_edges_from(model.bqm.quadratic)
    with DWaveSampler(solver={"name": name}) as sampler:
        accepted = valid_embeddings(banks[name], source, sampler.to_networkx_graph())
        if not accepted:
            raise RuntimeError(
                f"No saved embeddings remain valid on {name}; refresh QPU availability."
            )
        embedding = accepted[0]
        composite = FixedEmbeddingComposite(
            sampler, embedding.chains, source_adjacency=source.adj
        )
        samples = composite.sample(
            model.bqm,
            num_reads=count,
            answer_mode="raw",
            **model.sampling,
        )
        samples.resolve()
        samples = samples.change_vartype("SPIN", inplace=False)
        positions = [samples.variables.index(node) for node in model.nodes]
        spins = torch.from_numpy(samples.record.sample[:, positions].copy()).float()
        occurrences = torch.from_numpy(samples.record.num_occurrences.copy()).long()
        spins = spins.repeat_interleave(occurrences, dim=0)
        if spins.shape != (count, len(model.nodes)):
            raise RuntimeError(f"QPU returned {len(spins)} samples; expected {count}.")
        metadata = {
            "solver": name,
            "embedding_id": embedding.embedding_id,
            "valid_embeddings": len(accepted),
            "timing": samples.info.get("timing", {}),
        }
    return spins, metadata
