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

import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch
from torch import nn

from src.generation import generate_images, image_data_uri


class CountingDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.marker = nn.Parameter(torch.tensor(0.0))
        self.cfg = SimpleNamespace(img_size=2)
        self.decoded = 0

    def forward(self, spins):
        self.decoded += len(spins)
        return spins[:, :1, None, None].expand(-1, 3, 2, 2)


class GenerationTests(unittest.TestCase):
    def test_streaming_top_ten_scores_every_candidate_once(self):
        decoder = CountingDecoder()
        quality = Mock()
        quality.score.side_effect = lambda images: images[:, 0, 0, 0]
        diversity = Mock()
        diversity.pairwise.return_value = torch.ones(10, 10) - torch.eye(10)
        values = torch.tensor(
            [0.3, 0.9, 0.2, 0.8, 0.7, 0.5, 0.4, 0.1, 0.6, 1.0, 0.0, 0.95]
        )
        with patch("src.generation.select_diverse_indices", return_value=[0, 3, 6, 9]):
            result = generate_images(
                values[:, None], SimpleNamespace(decoder=decoder), quality, diversity, 5
            )
        expected = values.argsort(descending=True, stable=True)[:10]
        retained = diversity.pairwise.call_args.args[0][:, 0, 0, 0]
        torch.testing.assert_close(retained, values[expected])
        self.assertEqual(decoder.decoded, len(values))
        self.assertEqual(
            sum(len(call.args[0]) for call in quality.score.call_args_list), len(values)
        )
        self.assertEqual(quality.score.call_count, 3)
        diversity.pairwise.assert_called_once()
        self.assertEqual(result.generated_indices, expected[[0, 3, 6, 9]].tolist())
        self.assertEqual(result.quality_ranks, [1, 4, 7, 10])
        self.assertEqual(result.images.shape, (4, 3, 2, 2))

    def test_quality_ties_keep_generated_order_across_batches(self):
        decoder = CountingDecoder()
        quality = SimpleNamespace(score=lambda images: torch.zeros(len(images)))
        diversity = SimpleNamespace(
            pairwise=lambda images: torch.ones(10, 10) - torch.eye(10)
        )
        result = generate_images(
            torch.linspace(0, 1, 13)[:, None],
            SimpleNamespace(decoder=decoder),
            quality,
            diversity,
            3,
        )
        self.assertEqual(result.generated_indices, [0, 1, 2, 3])

    def test_rejects_nonfinite_quality_instead_of_showing_substitutes(self):
        decoder = CountingDecoder()
        quality = SimpleNamespace(
            score=lambda images: torch.full((len(images),), float("nan"))
        )
        with self.assertRaisesRegex(RuntimeError, "finite score"):
            generate_images(
                torch.ones(10, 1), SimpleNamespace(decoder=decoder), quality, Mock(), 2
            )

    def test_image_encoding_is_browser_local_png(self):
        uri = image_data_uri(torch.zeros(3, 2, 2))
        self.assertTrue(uri.startswith("data:image/png;base64,iVBOR"))
