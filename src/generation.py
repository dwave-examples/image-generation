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

"""Bounded-memory generation: decode once, score once, retain only the best ten."""

import base64
from dataclasses import dataclass
from functools import lru_cache
from io import BytesIO
from threading import Lock
from time import perf_counter

import networkx as nx
import torch
from PIL import Image

from demo_configs import (
    CANDIDATE_COUNT,
    CPU_THREADS,
    DECODE_BATCH_SIZE,
    DEVICE,
    EMBEDDING_DIRECTORY,
    MODEL_DIRECTORY,
)
from src.model import TrainedModel, load_model
from src.qpu import discover_solvers, load_banks, sample_qpu
from src.selection import EDiffIQA, PerceptualDistance, select_diverse_indices

TOP_K = 10
DISPLAY_COUNT = 4
generation_lock = Lock()


@dataclass
class GenerationResult:
    images: torch.Tensor
    scores: torch.Tensor
    quality_ranks: list[int]
    generated_indices: list[int]
    metadata: dict


@torch.inference_mode()
def generate_images(
    spins: torch.Tensor,
    model: TrainedModel,
    quality: EDiffIQA,
    diversity: PerceptualDistance,
    batch_size: int,
) -> GenerationResult:
    if len(spins) < TOP_K or batch_size < 1:
        raise ValueError(
            "Generate at least ten candidates with a positive decoder batch size."
        )
    device = next(model.decoder.parameters()).device
    best_images = torch.empty(
        0, 3, model.decoder.cfg.img_size, model.decoder.cfg.img_size
    )
    best_scores = torch.empty(0)
    best_indices = torch.empty(0, dtype=torch.long)
    for start in range(0, len(spins), batch_size):
        images = model.decoder(spins[start : start + batch_size].to(device)).cpu()
        if not torch.isfinite(images).all():
            raise RuntimeError("The decoder returned non-finite image pixels.")
        scores = quality.score(images)
        if scores.shape != (len(images),) or not torch.isfinite(scores).all():
            raise RuntimeError("eDifFIQA must return one finite score per image.")
        all_scores = torch.cat((best_scores, scores))
        indices = torch.argsort(all_scores, descending=True, stable=True)[:TOP_K]
        best_images = torch.cat((best_images, images))[indices]
        best_scores = all_scores[indices]
        best_indices = torch.cat(
            (best_indices, torch.arange(start, start + len(images)))
        )[indices]
    distances = diversity.pairwise(best_images)
    selected = select_diverse_indices(distances, DISPLAY_COUNT)
    return GenerationResult(
        best_images[selected],
        best_scores[selected],
        [index + 1 for index in selected],
        best_indices[selected].tolist(),
        {
            "candidates": len(spins),
            "top_k": TOP_K,
            "pairwise_distances": distances.tolist(),
        },
    )


def image_data_uri(image: torch.Tensor) -> str:
    pixels = image.clamp(0, 1).permute(1, 2, 0).mul(255).round().byte().numpy()
    buffer = BytesIO()
    Image.fromarray(pixels).save(buffer, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode(
        "ascii"
    )


class DemoEngine:
    def __init__(self) -> None:
        if (
            CANDIDATE_COUNT < TOP_K
            or DECODE_BATCH_SIZE < 1
            or not 1 <= CPU_THREADS <= 4
        ):
            raise ValueError(
                "Invalid candidate count, decoder batch size, or CPU thread limit."
            )
        torch.set_num_threads(CPU_THREADS)
        self.device = torch.device(DEVICE)
        self.model = load_model(MODEL_DIRECTORY, self.device)
        self.banks = load_banks(EMBEDDING_DIRECTORY)
        self.source = nx.Graph()
        self.source.add_nodes_from(self.model.nodes)
        self.source.add_edges_from(self.model.bqm.quadratic)
        self.quality = EDiffIQA(CPU_THREADS)
        self.diversity = PerceptualDistance(self.device)

    def available_solvers(self) -> tuple[list[dict], list[str]]:
        return discover_solvers(self.banks, self.source)

    def generate(self, solver: str) -> GenerationResult:
        with generation_lock:
            started = perf_counter()
            spins, qpu_metadata = sample_qpu(
                solver, CANDIDATE_COUNT, self.model, self.banks
            )
            result = generate_images(
                spins, self.model, self.quality, self.diversity, DECODE_BATCH_SIZE
            )
            result.metadata.update(qpu_metadata)
            result.metadata.update(
                seconds=perf_counter() - started,
                checkpoint_step=self.model.source["step"],
            )
            return result


@lru_cache(maxsize=1)
def get_engine() -> DemoEngine:
    return DemoEngine()
