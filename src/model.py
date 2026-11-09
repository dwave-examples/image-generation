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

"""Load the portable, tensor-only EMA decoder and its logical Ising model."""

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import dimod
import torch

from src.ffhq_decoder import DecoderConfig, FFHQDecoder

MODEL_FORMAT = "ffhq_demo_inference_v1"


def file_sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


@dataclass
class TrainedModel:
    decoder: FFHQDecoder
    bqm: dimod.BinaryQuadraticModel
    nodes: list[int]
    source: dict
    sampling: dict


def load_model(directory: Path, device: torch.device) -> TrainedModel:
    manifest = json.loads((directory / "manifest.json").read_text())
    if manifest["format"] != MODEL_FORMAT:
        raise ValueError("Unsupported FFHQ inference model format.")
    tensors = {}
    for record in manifest["shards"]:
        path = directory / record["path"]
        if file_sha256(path) != record["sha256"]:
            raise ValueError(f"Model checksum mismatch: {path.name}")
        shard = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
        if tensors.keys() & shard.keys():
            raise ValueError(f"Duplicate model tensors in {path.name}.")
        tensors.update(shard)

    nodes = manifest["nodes"]
    edges = manifest["edges"]
    energy = tensors.pop("energy_vector")
    if energy.numel() != len(nodes) + len(edges) or not torch.isfinite(energy).all():
        raise ValueError("Invalid logical Ising energy vector.")
    h, j = energy[: len(nodes)], energy[len(nodes) :]
    positions = {node: index for index, node in enumerate(nodes)}
    coupling = torch.zeros(len(nodes), len(nodes))
    for (u, v), weight in zip(edges, j):
        left, right = positions[u], positions[v]
        coupling[left, right] = coupling[right, left] = weight
    decoder = FFHQDecoder(DecoderConfig(**manifest["decoder_config"]), coupling, h)
    decoder.load_state_dict(tensors, strict=True)
    if not decoder.qpu_global_mode_residual.basis_initialized.item():
        raise ValueError("The checkpoint is missing its fitted global-mode basis.")
    decoder = (
        decoder.to(device=device, memory_format=torch.channels_last)
        .eval()
        .requires_grad_(False)
    )
    bqm = dimod.BinaryQuadraticModel(
        dict(zip(nodes, h.tolist())),
        dict(zip(map(tuple, edges), j.tolist())),
        0.0,
        dimod.SPIN,
    )
    return TrainedModel(decoder, bqm, nodes, manifest["source"], manifest["sampling"])
