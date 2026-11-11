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

"""eDifFIQA ranking and diversity selection with one feature pass per image."""

from itertools import combinations
from pathlib import Path

import lpips
import torch
from huggingface_hub import hf_hub_download

QUALITY_REPO = "opencv/face_image_quality_assessment_ediffiqa"
QUALITY_FILE = "ediffiqa_tiny_jun2024.onnx"


class EDiffIQA:
    def __init__(self, cpu_threads: int = 4, model_path: Path | None = None) -> None:
        import cv2

        cv2.setNumThreads(cpu_threads)
        path = model_path or Path(hf_hub_download(QUALITY_REPO, QUALITY_FILE))
        self.cv2 = cv2
        self.model = cv2.dnn.readNetFromONNX(str(path))
        self.model.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        self.model.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

    def score(self, images: torch.Tensor) -> torch.Tensor:
        scores = []
        for image in images.detach().float().clamp(0, 1).cpu():
            rgb = image.permute(1, 2, 0).numpy()
            rgb = self.cv2.resize(rgb, (112, 112), interpolation=self.cv2.INTER_LINEAR)
            normalized = ((rgb - 0.5) / 0.5).transpose(2, 0, 1)[None].astype("float32")
            self.model.setInput(normalized)
            scores.append(float(self.model.forward().squeeze()))
        scores = torch.tensor(scores)
        if not torch.isfinite(scores).all():
            raise RuntimeError("eDifFIQA returned non-finite quality scores.")
        return scores


class PerceptualDistance:
    def __init__(self, device: torch.device) -> None:
        self.model = (
            lpips.LPIPS(net="alex", verbose=False)
            .to(device)
            .eval()
            .requires_grad_(False)
        )
        self.device = device

    @torch.inference_mode()
    def pairwise(self, images: torch.Tensor) -> torch.Tensor:
        values = images.to(self.device).mul(2).sub(1)
        features = self.model.net(self.model.scaling_layer(values))
        pairs = torch.combinations(torch.arange(len(images), device=self.device))
        distances = torch.zeros(len(pairs), device=self.device)
        for feature, linear in zip(features, self.model.lins):
            normalized = lpips.normalize_tensor(feature)
            difference = normalized[pairs[:, 0]] - normalized[pairs[:, 1]]
            distances += linear(difference.square()).mean(dim=(1, 2, 3))
        matrix = torch.zeros(len(images), len(images), device=self.device)
        matrix[pairs[:, 0], pairs[:, 1]] = distances
        return (matrix + matrix.t()).cpu()


def select_diverse_indices(distances: torch.Tensor, count: int = 4) -> list[int]:
    """Maximize the closest pair, breaking ties by total distance then quality order."""
    if distances.ndim != 2 or distances.shape[0] != distances.shape[1]:
        raise ValueError("Perceptual distances must be a square matrix.")
    if not 2 <= count <= len(distances):
        raise ValueError("Selection count must be between two and the candidate count.")
    if not torch.isfinite(distances).all() or not torch.allclose(
        distances, distances.t()
    ):
        raise ValueError("Perceptual distances must be finite and symmetric.")
    best_indices = None
    best_score = None
    for indices in combinations(range(len(distances)), count):
        pair_distances = [float(distances[a, b]) for a, b in combinations(indices, 2)]
        score = (min(pair_distances), sum(pair_distances))
        if best_score is None or score > best_score:
            best_indices, best_score = indices, score
    return list(best_indices)
