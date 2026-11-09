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

"""Export an existing ml-etc checkpoint, without importing or including training code."""

import argparse
import json
from dataclasses import fields
from pathlib import Path

import dwave_networkx as dnx
import torch

from src.decoder import DecoderConfig
from src.model import MODEL_FORMAT, file_sha256

DECODER_PREFIXES = (
    "generator.blocks.",
    "quantize.embed.",
    "qpu_global_mode_residual.",
)


def export_model(checkpoint: Path, banks: list[Path], destination: Path) -> None:
    """Write tensor-only shards, explicit graph order, and portable embedding banks."""
    if destination.exists():
        raise FileExistsError(f"Refusing to overwrite an existing model: {destination}")
    # Only use a trusted training checkpoint; the resulting shards load with weights_only=True.
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False, mmap=True)
    cfg, run = payload["binaryae_config"], payload["run_config"]
    required = {
        "qpu_global_mode_residual": True,
        "qpu_global_mode_fusion": "normalized_add",
        "qpu_global_mode_basis_space": "decoder_input",
        "spin_latent": True,
        "qpu_gnn_decoder_transform": False,
        "qpu_latent_ntn_adapter": False,
        "qpu_ntn_adapter": False,
    }
    if any(cfg.get(key) != value for key, value in required.items()):
        raise ValueError(
            "This exporter only supports the current normalized-add FFHQ architecture."
        )
    if (
        str(run["qpu_decoder_input"]).lower() != "sh"
        or not run["qpu_fast_anneal"]
        or run["qpu_fast_anneal_use_linear_ancilla"]
    ):
        raise ValueError(
            "Expected sH decoder inputs and fast annealing without linear ancillas."
        )
    energy = (
        torch.as_tensor(payload["qpu_state"]["energy_vector"]).detach().cpu().clone()
    )
    source_info = None
    portable_banks = {}
    for bank in banks:
        manifest_path = bank / "manifest.json" if bank.is_dir() else bank
        manifest = json.loads(manifest_path.read_text())
        graph_info = manifest["source"]
        source = (graph_info["m"], graph_info["t"])
        if source_info is not None and source != source_info:
            raise ValueError(
                "All embedding banks must use the same logical source graph."
            )
        source_info = source
        records = []
        for record in manifest["embeddings"]:
            if not record.get("accepted", True):
                continue
            embedding = json.loads((manifest_path.parent / record["path"]).read_text())[
                "embedding"
            ]
            records.append(
                {"embedding_id": record["embedding_id"], "embedding": embedding}
            )
        if not records:
            raise ValueError(f"No accepted embeddings in {manifest_path}.")
        name = manifest["solver_name"]
        if name in portable_banks:
            raise ValueError(f"Duplicate embedding bank for {name}.")
        portable_banks[name] = {"solver_name": name, "embeddings": records}
    if source_info is None:
        raise ValueError("Supply at least one embedding bank.")
    graph = dnx.zephyr_graph(*source_info)
    nodes, edges = list(graph.nodes), list(graph.edges)
    if len(energy) != len(nodes) + len(edges) or not torch.isfinite(energy).all():
        raise ValueError("The energy vector does not match the embedding source graph.")
    if energy[: len(nodes)].count_nonzero():
        raise ValueError("Fast annealing requires zero logical linear biases.")
    state = {
        key.removeprefix("ae."): value.detach().cpu().clone()
        for key, value in payload["ema_model_state_dict"].items()
        if key.removeprefix("ae.").startswith(DECODER_PREFIXES)
    }
    if not state["qpu_global_mode_residual.basis_initialized"].item():
        raise ValueError("The checkpoint's global-mode basis has not been fitted.")
    state["energy_vector"] = energy
    destination.mkdir(parents=True)
    shards, shard, shard_bytes = [], {}, 0

    def save_shard() -> None:
        path = destination / f"decoder-{len(shards) + 1:02d}.pt"
        torch.save(shard, path)
        shards.append({"path": path.name, "sha256": file_sha256(path)})

    for key, value in state.items():
        size = value.numel() * value.element_size()
        if shard and shard_bytes + size > 64 * 1024**2:
            save_shard()
            shard, shard_bytes = {}, 0
        shard[key] = value
        shard_bytes += size
    if shard:
        save_shard()
    config = {field.name: cfg[field.name] for field in fields(DecoderConfig)}
    manifest = {
        "format": MODEL_FORMAT,
        "source": {
            "repository": "dwavesystems/ml-etc",
            "run": checkpoint.parent.name,
            "checkpoint": checkpoint.name,
            "sha256": file_sha256(checkpoint),
            "step": int(payload["step"]),
            "weights": "ema_model_state_dict",
        },
        "decoder_config": config,
        "nodes": nodes,
        "edges": edges,
        "sampling": {
            "fast_anneal": True,
            "annealing_time": float(run["qpu_fast_anneal_time_us"]),
        },
        "shards": shards,
    }
    (destination / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    embedding_dir = destination / "embeddings"
    embedding_dir.mkdir()
    for name, bank in portable_banks.items():
        (embedding_dir / f"{name}.json").write_text(json.dumps(bank, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--embedding-banks", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    export_model(args.checkpoint, args.embedding_banks, args.output)
