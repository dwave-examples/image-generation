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

import dimod
import networkx as nx
import torch

from src.qpu import SavedEmbedding, discover_solvers, sample_qpu, valid_embeddings


class QPUTests(unittest.TestCase):
    def setUp(self):
        self.source = nx.path_graph([8, 2])
        self.target = nx.path_graph([10, 11, 12])
        self.invalid = SavedEmbedding(0, {8: [90], 2: [91]})
        self.valid = SavedEmbedding(1, {8: [10], 2: [11]})

    def test_invalid_embedding_falls_back_to_saved_valid_embedding(self):
        with self.assertLogs("src.qpu", level="WARNING"):
            accepted = valid_embeddings(
                [self.invalid, self.valid], self.source, self.target
            )
        self.assertEqual(accepted, [self.valid])

    def test_missing_coupler_and_broken_chains_are_rejected(self):
        bad = [
            SavedEmbedding(0, {8: [10], 2: [12]}),
            SavedEmbedding(1, {8: [10, 12], 2: [11]}),
            SavedEmbedding(2, {8: [10]}),
        ]
        with self.assertLogs("src.qpu", level="WARNING"):
            self.assertEqual(valid_embeddings(bad, self.source, self.target), [])

    @patch("src.qpu.Client")
    def test_only_accessible_solvers_with_valid_embeddings_are_listed(self, client):
        client.from_config.return_value.__enter__.return_value.get_solvers.return_value = [
            SimpleNamespace(
                name="valid", nodes=self.target.nodes, edges=self.target.edges
            ),
            SimpleNamespace(
                name="invalid", nodes=self.target.nodes, edges=self.target.edges
            ),
        ]
        with self.assertLogs("src.qpu", level="WARNING"):
            options, unavailable = discover_solvers(
                {
                    "valid": [self.valid],
                    "invalid": [self.invalid],
                    "missing": [self.valid],
                },
                self.source,
            )
        self.assertEqual([option["value"] for option in options], ["valid"])
        self.assertEqual(len(unavailable), 2)
        client.from_config.assert_called_once_with(client="qpu")

    @patch("src.qpu.FixedEmbeddingComposite")
    @patch("src.qpu.DWaveSampler")
    def test_sampling_preserves_logical_order_and_expands_occurrences(
        self, sampler_class, composite
    ):
        sampler = sampler_class.return_value.__enter__.return_value
        sampler.to_networkx_graph.return_value = self.target
        samples = dimod.SampleSet.from_samples(
            ({2: 1, 8: -1}), vartype="SPIN", energy=[0], num_occurrences=[3]
        )
        composite.return_value.sample.return_value = samples
        model = SimpleNamespace(
            nodes=[8, 2],
            bqm=dimod.BinaryQuadraticModel({8: 0, 2: 0}, {(8, 2): -1}, 0, "SPIN"),
            sampling={"fast_anneal": True, "annealing_time": 0.1},
        )
        with self.assertLogs("src.qpu", level="WARNING"):
            spins, info = sample_qpu(
                "test", 3, model, {"test": [self.invalid, self.valid]}
            )
        torch.testing.assert_close(spins, torch.tensor([[-1.0, 1.0]] * 3))
        self.assertEqual(info["embedding_id"], 1)
        composite.return_value.sample.assert_called_once_with(
            model.bqm,
            num_reads=3,
            answer_mode="raw",
            fast_anneal=True,
            annealing_time=0.1,
        )
        self.assertEqual(composite.call_args.args[1], self.valid.chains)

    @patch("src.qpu.DWaveSampler")
    def test_qpu_errors_propagate_without_substitute_sampling(self, sampler):
        sampler.side_effect = RuntimeError("QPU unavailable")
        model = SimpleNamespace(
            nodes=[8, 2],
            bqm=dimod.BinaryQuadraticModel({8: 0, 2: 0}, {(8, 2): -1}, 0, "SPIN"),
        )
        with self.assertRaisesRegex(RuntimeError, "QPU unavailable"):
            sample_qpu("test", 128, model, {"test": [self.valid]})
