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

import lpips
import torch

from src.selection import PerceptualDistance, select_diverse_indices


class SelectionTests(unittest.TestCase):
    def test_exact_max_min_is_not_anchored_to_best_quality(self):
        distances = torch.full((5, 5), 10.0)
        distances.fill_diagonal_(0)
        distances[0, 1:] = 1
        distances[1:, 0] = 1
        self.assertEqual(select_diverse_indices(distances), [1, 2, 3, 4])

    def test_equal_diversity_prefers_quality_order(self):
        self.assertEqual(
            select_diverse_indices(torch.ones(10, 10) - torch.eye(10)), [0, 1, 2, 3]
        )

    def test_total_distance_breaks_max_min_ties(self):
        distances = torch.ones(5, 5) - torch.eye(5)
        distances[3, 4] = distances[4, 3] = 4
        self.assertEqual(select_diverse_indices(distances), [0, 1, 3, 4])

    def test_rejects_invalid_distance_matrices(self):
        for matrix in (
            torch.zeros(3, 4),
            torch.full((4, 4), float("nan")),
            torch.tril(torch.ones(4, 4)),
        ):
            with self.subTest(matrix=matrix):
                with self.assertRaises(ValueError):
                    select_diverse_indices(matrix)

    def test_cached_features_match_lpips_and_run_backbone_once(self):
        torch.manual_seed(4)
        # Random backbone avoids network downloads; the feature/distance path is identical.
        model = lpips.LPIPS(net="alex", pnet_rand=True, verbose=False).eval()
        distance = PerceptualDistance.__new__(PerceptualDistance)
        distance.model, distance.device = model, torch.device("cpu")
        images = torch.rand(4, 3, 64, 64)
        calls = []
        handle = model.net.register_forward_hook(lambda *_args: calls.append(1))
        actual = distance.pairwise(images)
        handle.remove()
        self.assertEqual(len(calls), 1)
        with torch.inference_mode():
            for left in range(4):
                for right in range(left + 1, 4):
                    expected = model(
                        images[left : left + 1],
                        images[right : right + 1],
                        normalize=True,
                    ).squeeze()
                    torch.testing.assert_close(
                        actual[left, right], expected, atol=1e-6, rtol=1e-5
                    )
