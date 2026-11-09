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

"""Inference-only FFHQ decoder, separate from the original MNIST decoder."""

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True)
class DecoderConfig:
    img_size: int = 256
    n_channels: int = 3
    nf: int = 128
    ch_mult: tuple[int, ...] = (1, 1, 2, 2, 4)
    res_blocks: int = 2
    gen_mul: float = 1.0
    attn_resolutions: tuple[int, ...] = (16,)
    codebook_size: int = 16
    emb_dim: int = 256
    norm_first: bool = True
    use_bottleneck_attention: bool = True
    qpu_global_mode_rank: int = 64

    @property
    def latent_resolution(self) -> int:
        return self.img_size // 2 ** (len(self.ch_mult) - 1)

    @property
    def latent_dim(self) -> int:
        return self.codebook_size * self.latent_resolution**2


def normalize(channels: int) -> nn.GroupNorm:
    return nn.GroupNorm(32, channels, eps=1e-6, affine=True)


def rms_normalize(values: torch.Tensor) -> torch.Tensor:
    return values / values.square().mean(dim=1, keepdim=True).clamp_min(1e-8).sqrt()


class Upsample(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(F.interpolate(x, scale_factor=2.0, mode="nearest"))


class ResBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.norm1 = normalize(in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.norm2 = normalize(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        # Retain this layer even for equal widths to match the saved state exactly.
        self.conv_out = nn.Conv2d(in_channels, out_channels, 1)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x if x.shape[1] == self.conv2.out_channels else self.conv_out(x)
        return residual + self.conv2(
            self.act(self.norm2(self.conv1(self.act(self.norm1(x)))))
        )


class AttnBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.norm = normalize(channels)
        self.q = nn.Conv2d(channels, channels, 1)
        self.k = nn.Conv2d(channels, channels, 1)
        self.v = nn.Conv2d(channels, channels, 1)
        self.proj_out = nn.Conv2d(channels, channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        normalized = self.norm(x)
        query = self.q(normalized).flatten(2).transpose(1, 2)
        key = self.k(normalized).flatten(2)
        value = self.v(normalized).flatten(2)
        weights = (query @ key * x.shape[1] ** -0.5).softmax(dim=2)
        attended = (value @ weights.transpose(1, 2)).reshape_as(x)
        return x + self.proj_out(attended)


class Generator(nn.Module):
    def __init__(self, cfg: DecoderConfig) -> None:
        super().__init__()
        channels = cfg.nf * cfg.ch_mult[-1]
        resolution = cfg.latent_resolution
        blocks = [normalize(cfg.emb_dim)] if cfg.norm_first else []
        blocks.extend(
            [
                nn.Conv2d(cfg.emb_dim, channels, 3, padding=1),
                ResBlock(channels, channels),
            ]
        )
        if cfg.use_bottleneck_attention:
            blocks.append(AttnBlock(channels))
        blocks.append(ResBlock(channels, channels))
        for level in reversed(range(len(cfg.ch_mult))):
            output_channels = cfg.nf * cfg.ch_mult[level]
            for _ in range(int(cfg.res_blocks * cfg.gen_mul)):
                blocks.append(ResBlock(channels, output_channels))
                channels = output_channels
                if resolution in cfg.attn_resolutions:
                    blocks.append(AttnBlock(channels))
            if level != 0:
                blocks.append(Upsample(channels))
                resolution *= 2
        blocks.extend(
            [normalize(channels), nn.Conv2d(channels, cfg.n_channels, 3, padding=1)]
        )
        self.blocks = nn.ModuleList(blocks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            x = block(x)
        return x


class GlobalModeResidual(nn.Module):
    """The checkpoint's fitted basis and normalized-add global-mode fusion."""

    def __init__(self, nodes: int, rank: int) -> None:
        super().__init__()
        self.mode_mlp = nn.Sequential(
            nn.Linear(rank, rank), nn.SiLU(), nn.Linear(rank, rank)
        )
        self.global_gate = nn.Parameter(torch.zeros(()))
        self.register_buffer("qpu_mean", torch.zeros(nodes))
        self.register_buffer("qpu_modes", torch.zeros(nodes, rank))
        self.register_buffer("basis_initialized", torch.tensor(False))

    def forward(self, local: torch.Tensor) -> torch.Tensor:
        coordinates = (local - self.qpu_mean) @ self.qpu_modes
        residual = self.mode_mlp(coordinates) @ self.qpu_modes.t()
        return rms_normalize(local) + self.global_gate.sigmoid() * rms_normalize(
            residual
        )


class FFHQDecoder(nn.Module):
    def __init__(
        self, cfg: DecoderConfig, coupling: torch.Tensor, bias: torch.Tensor
    ) -> None:
        super().__init__()
        self.cfg = cfg
        self.generator = Generator(cfg)
        self.quantize = nn.Module()
        self.quantize.embed = nn.Embedding(cfg.codebook_size, cfg.emb_dim)
        self.qpu_global_mode_residual = GlobalModeResidual(
            cfg.latent_dim, cfg.qpu_global_mode_rank
        )
        self.register_buffer("coupling", coupling, persistent=False)
        self.register_buffer("bias", bias, persistent=False)
        logical_nodes = coupling.shape[0]
        # Match ml-etc's floor-based 660 -> 4096 resizing, including the Hamiltonian.
        index = torch.arange(cfg.latent_dim) * logical_nodes // cfg.latent_dim
        self.register_buffer("latent_index", index, persistent=False)
        expanded = coupling.index_select(0, index).index_select(1, index)
        self.register_buffer(
            "expanded_coupling", expanded.to_sparse().coalesce(), persistent=False
        )

    def decoder_input(self, spins: torch.Tensor) -> torch.Tensor:
        resized = spins.index_select(1, self.latent_index)
        field = torch.sparse.mm(self.expanded_coupling, resized.t()).t()
        field = field + self.bias.index_select(0, self.latent_index)
        return self.qpu_global_mode_residual(resized * field)

    def forward(self, spins: torch.Tensor) -> torch.Tensor:
        code = self.decoder_input(spins).reshape(
            -1,
            self.cfg.codebook_size,
            self.cfg.latent_resolution,
            self.cfg.latent_resolution,
        )
        quantized = torch.einsum(
            "b n h w, n d -> b d h w", code, self.quantize.embed.weight
        )
        quantized = quantized.contiguous(memory_format=torch.channels_last)
        return self.generator(quantized).clamp(0, 1)
