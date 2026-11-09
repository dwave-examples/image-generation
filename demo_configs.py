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

"""Inference settings; no training or per-user model mutation is supported."""

from pathlib import Path

ROOT = Path(__file__).resolve().parent
MODEL_DIRECTORY = ROOT / "models" / "ffhq"
EMBEDDING_DIRECTORY = MODEL_DIRECTORY / "embeddings"

APP_TITLE = "Quantum FFHQ Image Generation"
DEFAULT_QPU = "Advantage2_system1"
DESCRIPTION = (
    "Generate new 256 x 256 faces using a pretrained FFHQ decoder and fresh QPU samples. "
    "The ten highest eDifFIQA-quality candidates are compared using perceptual image "
    "features, and the four most mutually dissimilar are displayed."
)

CANDIDATE_COUNT = 32
DECODE_BATCH_SIZE = 1
CPU_THREADS = 4
DEVICE = "cpu"
