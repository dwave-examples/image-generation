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

import dwave_networkx as dnx
import networkx as nx
import torch
from dwave.experimental.embedding_methods import zephyr_quotient_search
from dwave.system import DWaveSampler, FixedEmbeddingComposite
from minorminer import find_embedding
from minorminer.utils.parallel_embeddings import find_sublattice_embeddings


def greedy_get_subgraph(
    n_nodes: int,
    random_seed: Optional[int],
    graph: Optional[nx.Graph] = None,
    qpu: Optional[str] = None,
) -> nx.Graph:
    """Finds a subgraph of the QPU with latent space size number of nodes.

    Args:
        n_nodes: The number of nodes for the subgraph (equal to the latent space size).
        random_seed: The random seed for node selection.
        graph: The full QPU graph.
        qpu: The selected QPU.

    Returns:
        nx.Graph: The subgraph of the QPU with latent space size number of nodes.
    """
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

    return subgraph


def _try_zephyr_sublattice_fallback(
    n_nodes: int,
    qpu_graph: nx.Graph,
    random_seed: Optional[int],
) -> Optional[tuple[nx.Graph, dict]]:
    """Attempt to find a defect-free Zephyr sublattice embedding as a backup.

    Follows the approach demonstrated in
    https://github.com/dwavesystems/dwave-experimental/blob/main/examples/fully_yielded_zephyr_subgraph.py:
    1. Locate a complete Zephyr sublattice in the defective QPU graph via
       ``find_sublattice_embeddings``.
    2. Relabel it to canonical coordinates for ``zephyr_quotient_search``.
    3. Embed a source Zephyr graph (with ``n_nodes`` nodes) into the sublattice.
    4. Refine with ``find_embedding`` if quotient search did not reach full yield.
    5. Map the resulting chain embedding back to original QPU qubit labels.

    Args:
        n_nodes: Required number of logical nodes (equal to the latent space size).
        qpu_graph: The defective QPU graph as returned by ``qpu.to_networkx_graph()``.
        random_seed: Random seed forwarded to sublattice search.

    Returns:
        ``(logical_graph, chain_embedding)`` where ``logical_graph`` has integer
        nodes ``0..n_nodes-1`` and ``chain_embedding`` maps each integer node to a
        list of physical QPU qubits suitable for ``FixedEmbeddingComposite``.
        Returns ``None`` when no suitable embedding can be found.
    """
    qpu_rows = qpu_graph.graph.get("rows")
    qpu_tile = qpu_graph.graph.get("tile", 4)
    labels = qpu_graph.graph.get("labels")

    if not isinstance(qpu_rows, int) or qpu_rows <= 0:
        return None

    use_coords = labels == "coordinate"
    zephyr_tile = dnx.zephyr_graph(qpu_rows, qpu_tile, coordinates=use_coords)

    tile_embeddings = find_sublattice_embeddings(
        S=zephyr_tile,
        T=qpu_graph,
        max_num_emb=1,
        one_to_iterable=False,
        seed=random_seed,
    )
    if not tile_embeddings:
        print("Zephyr fallback: no complete sublattice found in defective QPU graph.")
        return None

    tile_embedding = tile_embeddings[0]  # {tile_node: physical_qubit}

    sublattice_nodes = set(tile_embedding.values())
    target_sub = qpu_graph.subgraph(sublattice_nodes).copy()
    inv_map = {phys: tile_node for tile_node, phys in tile_embedding.items()}
    target_sub = nx.relabel_nodes(target_sub, inv_map, copy=True)
    target_sub.graph.update(
        family="zephyr",
        rows=qpu_rows,
        tile=qpu_tile,
        labels="coordinate" if use_coords else "int",
    )

    source = None
    for t_s in range(qpu_tile, 0, -1):
        candidate = dnx.zephyr_graph(qpu_rows, t_s, coordinates=use_coords)
        if candidate.number_of_nodes() == n_nodes:
            source = candidate
            break

    if source is None:
        print(
            f"Zephyr fallback: no source Zephyr graph with {n_nodes} nodes found "
            f"for QPU rows={qpu_rows} tile={qpu_tile}."
        )
        return None

    emb, metadata = zephyr_quotient_search(source, target_sub, yield_type="edge")
    best_emb = emb

    if metadata.final_num_yielded < metadata.max_num_yielded:
        print(
            f"Zephyr fallback: quotient search placed {metadata.final_num_yielded}/"
            f"{metadata.max_num_yielded} edges; refining with find_embedding."
        )
        refined = find_embedding(S=source, T=target_sub, initial_chains=emb, timeout=50)
        if refined:
            best_emb = refined

    if not best_emb:
        print("Zephyr fallback: embedding search returned no result.")
        return None

    chain_embedding_in_qpu = {
        s_node: [tile_embedding[v] for v in chain]
        for s_node, chain in best_emb.items()
    }

    source_to_int = {node: i for i, node in enumerate(source.nodes())}
    logical_graph = nx.relabel_nodes(source, source_to_int)
    int_chain_embedding = {
        source_to_int[s]: chains for s, chains in chain_embedding_in_qpu.items()
    }

    return logical_graph, int_chain_embedding


def get_graph_mapping(graph: Optional[nx.Graph]) -> tuple[nx.Graph, dict]:
    """Maps a graph of the QPU to the encoded latent data.

    Args:
        graph: The graph of the QPU with latent space size number of nodes.

    Returns:
        nx.Graph: The input graph with qubit nodes mapped to ints from 0 to len(graph)
        dict: A mapping of qubits to integers from 0 to len(graph)
    """
    mapping = {
        physical: logical for (physical, logical) in zip(graph.nodes(), range(len(graph)))
    }

    return nx.relabel_nodes(graph, mapping), mapping


def get_sampler_and_sampler_kwargs(
    num_reads: int, annealing_time: float, n_latents: int, random_seed: int, qpu: str
) -> tuple[FixedEmbeddingComposite, dict, nx.Graph, tuple[float, float], tuple[float, float]]:
    """TODO

    Args:
        num_reads: TODO
        annealing_time: TODO
        n_latents: TODO
        random_seed: TODO
        qpu: TODO

    Returns:
        sampler: TODO
        sampler_kwargs: TODO
        graph: TODO
        linear_range: TODO
        quadratic_range: TODO
    """

    qpu = DWaveSampler(solver=qpu)
    qpu_graph = qpu.to_networkx_graph()

    try:
        subgraph = greedy_get_subgraph(n_nodes=n_latents, random_seed=random_seed, graph=qpu_graph)
        mapped_graph, mapping = get_graph_mapping(subgraph)
        sampler = FixedEmbeddingComposite(qpu, {l_: [p] for p, l_ in mapping.items()})
    except Exception:
        qpu_topology = qpu.properties["topology"]["type"]
        if qpu_topology != "zephyr":
            raise
        fallback = _try_zephyr_sublattice_fallback(
            n_nodes=n_latents, qpu_graph=qpu_graph, random_seed=random_seed
        )
        if fallback is None:
            raise
        mapped_graph, chain_embedding = fallback
        sampler = FixedEmbeddingComposite(qpu, chain_embedding)

    linear_range, quadratic_range = qpu.properties["h_range"], qpu.properties["j_range"]
    sampler_kwargs = dict(
        num_reads=num_reads,
        # Set `answer_mode` to "raw" so no samples are aggregated
        answer_mode="raw",
        # Set `auto_scale`` to `False` so the sampler sample from the intended distribution
        auto_scale=False,
        annealing_time=annealing_time,
        label="Examples - ML MNIST Image Gen",
    )

    return sampler, sampler_kwargs, mapped_graph, linear_range, quadratic_range


def get_latent_to_discrete(
    mode: Optional[Literal["heaviside"]],
) -> Optional[Callable[[torch.Tensor, int], torch.Tensor]]:
    """TODO

    Args:
        mode: TODO

    Returns:
        latent_to_discrete: TODO
    """
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
