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

from typing import TYPE_CHECKING, Optional

import torch
from dimod import Sampler, SampleSet

if TYPE_CHECKING:
    from dwave.plugins.torch.models import GraphRestrictedBoltzmannMachine
    from .utils.persistent_qpu_sampler import PersistentQPUSampleHelper

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
