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

import torch

from demo_configs import MODEL_DIRECTORY
from src.ffhq_decoder import DecoderConfig, FFHQDecoder, rms_normalize
from src.model import load_model


class DecoderTests(unittest.TestCase):
    def test_local_hamiltonian_resizing_and_global_mode_fusion(self):
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
        coupling = torch.tensor([[0.0, 2.0, 1.0], [2.0, 0.0, -1.0], [1.0, -1.0, 0.0]])
        bias = torch.tensor([0.1, 0.2, 0.3])
        model = FFHQDecoder(cfg, coupling, bias)
        modes = model.qpu_global_mode_residual
        modes.qpu_modes.copy_(torch.randn(cfg.latent_dim, 2))
        modes.qpu_mean.copy_(torch.randn(cfg.latent_dim))
        spins = torch.tensor([[-1.0, 1.0, -1.0], [1.0, -1.0, 1.0]])
        index = torch.arange(cfg.latent_dim) * 3 // cfg.latent_dim
        expanded = coupling[index][:, index]
        resized = spins[:, index]
        local = resized * (resized @ expanded.t() + bias[index])
        residual = (
            modes.mode_mlp((local - modes.qpu_mean) @ modes.qpu_modes)
            @ modes.qpu_modes.t()
        )
        expected = rms_normalize(local) + modes.global_gate.sigmoid() * rms_normalize(
            residual
        )
        torch.testing.assert_close(model.decoder_input(spins), expected)

    def test_bundled_checkpoint_loads_strictly_and_decodes_on_cpu(self):
        torch.set_num_threads(2)
        model = load_model(MODEL_DIRECTORY, torch.device("cpu"))
        with torch.inference_mode():
            images = model.decoder(torch.ones(1, len(model.nodes)))
        self.assertEqual(model.source["step"], 540000)
        self.assertEqual(images.shape, (1, 3, 256, 256))
        self.assertTrue(torch.isfinite(images).all())
        self.assertGreaterEqual(float(images.min()), 0)
        self.assertLessEqual(float(images.max()), 1)
