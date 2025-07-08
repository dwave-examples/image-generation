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
import torch
from dwave.plugins.torch.boltzmann_machine import GraphRestrictedBoltzmannMachine
from dimod import Sampler, SampleSet
from dwave.plugins.torch.utils import sample_to_tensor, spread

from .utils.persistent_qpu_sampler import PersistentQPUSampleHelper


class RadialBasisFunction(torch.nn.Module):
    def __init__(
        self,
        num_features: int,
        mul_factor: int | float = 2.0,
        bandwidth: float | None = None,
    ):
        super().__init__()
        bandwidth_multipliers = mul_factor ** (
            torch.arange(num_features) - num_features // 2
        )
        self.register_buffer("bandwidth_multipliers", bandwidth_multipliers)
        self.bandwidth = bandwidth

    def get_bandwidth(
        self, l2_distance_matrix: torch.Tensor | None = None
    ) -> torch.Tensor | float:
        if self.bandwidth is None:
            assert l2_distance_matrix is not None
            num_samples = l2_distance_matrix.shape[0]
            return l2_distance_matrix.sum() / (num_samples * (num_samples - 1))
        return self.bandwidth

    def forward(self, x):
        distance_matrix = torch.cdist(x, x, p=2)
        bandwidth = (
            self.get_bandwidth(distance_matrix.detach()) * self.bandwidth_multipliers
        )
        return torch.exp(
            -distance_matrix.unsqueeze(0) / bandwidth.reshape(-1, 1, 1)
        ).sum(dim=0)


def mmd_loss(
    spins: torch.Tensor,
    kernel: RadialBasisFunction,
    grbm: GraphRestrictedBoltzmannMachine,
    sampler: Sampler,
    sampler_kwargs: dict,
    prefactor: float,
):
    with torch.no_grad():
        grbm.h.mul_(prefactor)
        grbm.J.mul_(prefactor)
        h, J = grbm.ising
        sample_set = sampler.sample_ising(h, J, **sampler_kwargs)
        grbm.h.mul_(1 / prefactor)
        grbm.J.mul_(1 / prefactor)
        h, J = grbm.ising

    samples = sample_to_tensor(spread(sample_set)).to(spins.device)
    spins = spins.reshape(-1, spins.shape[-1])
    kernel_matrix = kernel(torch.vstack((spins, samples)))
    num_spin_strings = spins.shape[0]
    spin_spin_kernels = kernel_matrix[:num_spin_strings, :num_spin_strings]
    sample_sample_kernels = kernel_matrix[num_spin_strings:, num_spin_strings:]
    spin_sample_kernels = kernel_matrix[:num_spin_strings, num_spin_strings:]
    mmd = (
        spin_spin_kernels.mean()
        - 2 * spin_sample_kernels.mean()
        + sample_sample_kernels.mean()
    )
    return mmd


def nll_loss(
    spins: torch.Tensor,
    grbm: GraphRestrictedBoltzmannMachine,
    sampler: Sampler,
    sampler_kwargs: dict,
    prefactor: float,
    measure_prefactor: bool,
    persistent_qpu_sample_helper: PersistentQPUSampleHelper,
    sample_set: SampleSet | None = None,
) -> tuple[float, torch.Tensor, SampleSet]:
    prefactor, sample_set = persistent_qpu_sample_helper.find_prefactor(
        prefactor,
        grbm,
        sampler,
        sampler_kwargs,
        measure_prefactor=measure_prefactor,
        resample=False,
        reset_deque=True,
    )
    samples = sample_to_tensor(spread(sample_set)).to(spins.device)
    spins = spins.reshape(-1, spins.shape[-1])
    nll = torch.mean(grbm(spins)) - torch.mean(grbm(samples))
    return prefactor, nll, sample_set
