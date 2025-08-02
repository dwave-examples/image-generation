# Copyright 2025 D-Wave
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
#
# The use of the Maximum Mean Discrepancy (MMD) loss implementations
# below (including the mmd_loss function) with a quantum computing
# system is protected by the intellectual property rights of D-Wave
# Quantum Inc. and its affiliates.
#
# The use of the Maximum Mean Discrepancy (MMD) loss implementations
# below (including the mmd_loss function) with D-Wave's quantum computing
# system will require access to D-Wave’s LeapTM quantum cloud service and
# will be governed by the Leap Cloud Subscription Agreement available at:
# https://cloud.dwavesys.com/leap/legal/cloud_subscription_agreement/
#

from __future__ import annotations

from typing import TYPE_CHECKING, Optional, Union

import torch
from dimod import Sampler, SampleSet

if TYPE_CHECKING:
    from dwave.plugins.torch.models import GraphRestrictedBoltzmannMachine
    from .utils.persistent_qpu_sampler import PersistentQPUSampleHelper


class RadialBasisFunction(torch.nn.Module):
    """Radial basis function with multiple bandwidth parameters."""

    def __init__(
        self,
        num_features: int,
        mul_factor: Union[int, float] = 2.0,
        bandwidth: Optional[float] = None,
    ):
        super().__init__()
        bandwidth_multipliers = mul_factor ** (torch.arange(num_features) - num_features // 2)
        self.register_buffer("bandwidth_multipliers", bandwidth_multipliers)
        self.bandwidth = bandwidth

    def get_bandwidth(
        self,
        l2_distance_matrix: Optional[torch.Tensor] = None
    ) -> Union[torch.Tensor, float]:
        """A heuristic method for determining an appropriate bandwidth for the radial basis function."""
        if self.bandwidth is None:
            assert l2_distance_matrix is not None

            num_samples = l2_distance_matrix.shape[0]

            return l2_distance_matrix.sum() / (num_samples * (num_samples - 1))

        return self.bandwidth

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        distance_matrix = torch.cdist(x, x, p=2)
        bandwidth = self.get_bandwidth(distance_matrix.detach()) * self.bandwidth_multipliers

        return torch.exp(-distance_matrix.unsqueeze(0) / bandwidth.reshape(-1, 1, 1)).sum(dim=0)


def mmd_loss(
    spins: torch.Tensor,
    kernel: RadialBasisFunction,
    grbm: GraphRestrictedBoltzmannMachine,
    sampler: Sampler,
    sampler_kwargs: dict,
    linear_range: tuple[float, float],
    quadratic_range: tuple[float, float],
    prefactor: float,
) -> float:
    """Computes an unbiased estimate of the maximum mean discrepancy metric."""
    with torch.no_grad():
        samples = grbm.sample(
            sampler,
            prefactor=prefactor,
            device=spins.device,
            linear_range=linear_range,
            quadratic_range=quadratic_range,
            sample_params=sampler_kwargs,
        )

    spins = spins.reshape(-1, spins.shape[-1])

    kernel_matrix = kernel(torch.vstack((spins, samples)))
    num_spin_strings = spins.shape[0]
    spin_spin_kernels = kernel_matrix[:num_spin_strings, :num_spin_strings]
    sample_sample_kernels = kernel_matrix[num_spin_strings:, num_spin_strings:]
    spin_sample_kernels = kernel_matrix[:num_spin_strings, num_spin_strings:]

    mmd = spin_spin_kernels.mean() - 2 * spin_sample_kernels.mean() + sample_sample_kernels.mean()

    return mmd


def nll_loss(
    spins: torch.Tensor,
    grbm: GraphRestrictedBoltzmannMachine,
    sampler: Sampler,
    sampler_kwargs: dict,
    linear_range: tuple[float, float],
    quadratic_range: tuple[float, float],
    prefactor: float,
    persistent_qpu_sample_helper: PersistentQPUSampleHelper,
    sample_set: Optional[SampleSet] = None,
) -> tuple[float, torch.Tensor, SampleSet]:
    """A quasi-objective function with gradients equivalent to that of the negative log-likelihood of data."""
    sample_set = persistent_qpu_sample_helper.sample(
        prefactor,
        grbm,
        sampler,
        sampler_kwargs,
        linear_range,
        quadratic_range,
    )

    samples = grbm.sampleset_to_tensor(sample_set, device=spins.device)
    spins = spins.reshape(-1, spins.shape[-1])
    nll = torch.mean(grbm(spins)) - torch.mean(grbm(samples))

    return nll, sample_set
