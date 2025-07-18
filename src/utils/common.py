# Copyright 2025 D-Wave
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
import random
from typing import Callable, Literal, Optional

import networkx as nx
import torch
from dwave.system import DWaveSampler, FixedEmbeddingComposite


def greedy_get_subgraph(
    n_nodes: int,
    random_seed: Optional[int],
    graph: Optional[nx.Graph] = None,
    qpu: Optional[str] = None
) -> tuple[nx.Graph, dict]:
    """TODO"""
    generator = random.Random(random_seed)
    if graph is None:
        qpu = DWaveSampler(solver=qpu)
        graph = qpu.to_networkx_graph()

    assert isinstance(graph, nx.Graph)

    selected_nodes = [generator.choice(list(graph.nodes()))]
    MAXIMUM_CONNECTIVITY = max([len(list(graph.neighbors(n))) for n in graph.nodes()])

    while len(selected_nodes) < n_nodes:
        maximum_connectivity = 0
        target_maximum_connectivity = min(MAXIMUM_CONNECTIVITY, len(selected_nodes))
        target_node = None
        found_optimal_target_node = False
        generator.shuffle(selected_nodes)

        for node in selected_nodes:
            neighbours = list(graph.neighbors(node))
            generator.shuffle(neighbours)

            for neighbour in neighbours:
                if neighbour not in selected_nodes:
                    # If we were to add neighbour to the selected nodes, how many of the
                    # selected nodes would it be connected to?
                    connectivity = len(set(graph.neighbors(neighbour)).intersection(selected_nodes))
                    if connectivity >= target_maximum_connectivity:
                        found_optimal_target_node = True
                        target_node = neighbour
                        selected_nodes.append(target_node)
                        break
                    elif connectivity > maximum_connectivity:
                        maximum_connectivity = connectivity
                        target_node = neighbour

            if found_optimal_target_node:
                break

        if found_optimal_target_node:
            continue

        selected_nodes.append(target_node)

    subgraph = graph.subgraph(selected_nodes)
    mapping = {
        physical: logical for (physical, logical) in zip(subgraph.nodes(), range(len(subgraph)))
    }

    return nx.relabel_nodes(subgraph, mapping), mapping


def get_sampler_and_sampler_kwargs(num_reads, annealing_time, n_latents, random_seed, qpu: Optional[str] = None):
    """TODO: switch to work with refactored plugin"""
    qpu = DWaveSampler(solver=qpu)
    graph = qpu.to_networkx_graph()
    graph, mapping = greedy_get_subgraph(n_nodes=n_latents, random_seed=random_seed, graph=graph)

    if qpu:
        sampler = FixedEmbeddingComposite(qpu, {l_: [p] for p, l_ in mapping.items()})
        linear_range, quadratic_range = qpu.properties["h_range"], qpu.properties["j_range"]
        sampler_kwargs = dict(
            num_reads=num_reads,
            # Set `answer_mode` to "raw" so no samples are aggregated
            answer_mode="raw",
            # Set `auto_scale`` to `False` so the sampler sample from the intended distribution
            auto_scale=False,
            annealing_time=annealing_time,
            label="Examples - DVAE",
        )

    else:
        from dwave.samplers import SimulatedAnnealingSampler

        sampler = SimulatedAnnealingSampler()
        linear_range, quadratic_range = None, None
        sampler_kwargs = dict(
            num_reads=num_reads,
            beta_range=[1, 1],
            proposal_acceptance_criterion="Gibbs",
            randomize_order=True,
            num_sweeps=100,
        )

    return sampler, sampler_kwargs, graph, linear_range, quadratic_range


def get_latent_to_discrete(
    mode: Optional[Literal["heaviside"]],
) -> Callable[[torch.Tensor, int], torch.Tensor] | None:
    """TODO"""
    if mode is None:
        return None

    if mode != "heaviside":
        raise ValueError("Invalid Mode: Mode is not heaviside.")

    def latent_to_discrete(logits: torch.Tensor, n_samples: int) -> torch.Tensor:
        # logits is of shape (batch_size, n_discrete)
        # we ignore n_samples as we won't be doing any stochastic processes
        with torch.no_grad():
            hard = (
                torch.heaviside(
                    logits,
                    values=torch.tensor(0, device=logits.device, dtype=logits.dtype),
                )
                * 2
                - 1
            )

        return (hard - logits.detach() + logits).unsqueeze(1)

    return latent_to_discrete
