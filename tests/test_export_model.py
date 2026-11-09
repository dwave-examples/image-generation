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

import json
import tempfile
import unittest
from dataclasses import asdict
from pathlib import Path

import dwave_networkx as dnx
import torch

from src.decoder import DecoderConfig, FFHQDecoder
from src.export_model import export_model
from src.model import load_model


class ExportTests(unittest.TestCase):
    def test_export_roundtrip_keeps_inference_only_and_exact_ising_order(self):
        torch.set_num_threads(2)
        cfg = DecoderConfig(
            img_size=2,
            nf=32,
            ch_mult=(1,),
            emb_dim=32,
            codebook_size=2,
            qpu_global_mode_rank=2,
            use_bottleneck_attention=False,
            attn_resolutions=(),
        )
        graph = dnx.zephyr_graph(1, 1)
        nodes, edges = list(graph.nodes), list(graph.edges)
        energy = torch.cat(
            (torch.zeros(len(nodes)), torch.arange(len(edges)).float() / 100)
        )
        coupling = torch.zeros(len(nodes), len(nodes))
        for (u, v), weight in zip(edges, energy[len(nodes) :]):
            coupling[u, v] = coupling[v, u] = weight
        decoder = FFHQDecoder(cfg, coupling, energy[: len(nodes)])
        decoder.qpu_global_mode_residual.basis_initialized.fill_(True)
        payload = {
            "step": 42,
            "binaryae_config": {
                **asdict(cfg),
                "qpu_global_mode_residual": True,
                "qpu_global_mode_fusion": "normalized_add",
                "qpu_global_mode_basis_space": "decoder_input",
                "spin_latent": True,
                "qpu_gnn_decoder_transform": False,
                "qpu_latent_ntn_adapter": False,
                "qpu_ntn_adapter": False,
            },
            "run_config": {
                "qpu_decoder_input": "sH",
                "qpu_fast_anneal": True,
                "qpu_fast_anneal_use_linear_ancilla": False,
                "qpu_fast_anneal_time_us": 0.1,
                "training_only_setting": "not exported",
            },
            "ema_model_state_dict": {
                **{"ae." + key: value for key, value in decoder.state_dict().items()},
                "ae.encoder.weight": torch.ones(3),
                "discriminator.weight": torch.ones(3),
            },
            "qpu_state": {"energy_vector": energy},
            "optimizer_state_dict": {"unused": torch.ones(2)},
        }
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            checkpoint = root / "training.pt"
            torch.save(payload, checkpoint)
            bank = root / "bank"
            bank.mkdir()
            (bank / "embedding.json").write_text(
                json.dumps({"embedding": {str(node): [node] for node in nodes}})
            )
            (bank / "manifest.json").write_text(
                json.dumps(
                    {
                        "source": {"m": 1, "t": 1},
                        "solver_name": "test",
                        "embeddings": [{"embedding_id": 7, "path": "embedding.json"}],
                    }
                )
            )
            output = root / "export"
            export_model(checkpoint, [bank], output)
            model = load_model(output, torch.device("cpu"))
            self.assertEqual(model.nodes, nodes)
            for edge, weight in zip(edges, energy[len(nodes) :]):
                self.assertEqual(model.bqm.quadratic[edge], float(weight))
            spins = torch.ones(1, len(nodes))
            with torch.inference_mode():
                torch.testing.assert_close(model.decoder(spins), decoder(spins))
            manifest = json.loads((output / "manifest.json").read_text())
            self.assertNotIn("run_config", manifest)
            self.assertNotIn("optimizer_state_dict", manifest)
            with self.assertRaises(FileExistsError):
                export_model(checkpoint, [bank], output)
            shard = output / manifest["shards"][0]["path"]
            with shard.open("ab") as stream:
                stream.write(b"corruption")
            with self.assertRaisesRegex(ValueError, "checksum mismatch"):
                load_model(output, torch.device("cpu"))
