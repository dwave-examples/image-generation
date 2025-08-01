from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import torch
from dimod import Sampler, SampleSet, as_samples

if TYPE_CHECKING:
    from dwave.plugins.torch.boltzmann_machine import GraphRestrictedBoltzmannMachine


def push_to_deque(
    deque: torch.Tensor,
    x: torch.Tensor,
    deque_size: Optional[int] = None,
    dim: int = 0,
) -> torch.Tensor:
    """Handling `deque` tensor as a (set of) deque/FIFO, push the content of `x` into it."""
    if deque_size is None:
        deque_size = deque.shape[dim]

    deque_dims = deque.dim()
    input_size = x.shape[dim]
    dims_right = deque_dims - dim - 1
    deque_slicing = (
        (slice(None),) * dim
        + (
            slice(
                (input_size - deque_size if input_size < deque_size else deque.shape[dim]),
                None,
            ),
        )
        + (slice(None),) * dims_right
    )
    input_slicing = (slice(None),) * dim + (slice(-deque_size, None),) + (slice(None),) * dims_right
    deque = torch.cat((deque[deque_slicing], x[input_slicing]), dim=dim)

    return deque


class PersistentQPUSampleHelper:
    """A QPU wrapper that caches/reuses a sample."""

    def __init__(self, max_deque_size: int, iterations_before_resampling: int):
        self.current_deque_size = 0
        self.max_deque_size = max_deque_size
        self.iterations_before_resampling = iterations_before_resampling
        self.iterations_since_last_resampling = 0
        self.deque = None

    def sample(
        self,
        prefactor,
        grbm: GraphRestrictedBoltzmannMachine,
        sampler: Sampler,
        sampler_kwargs: dict,
        linear_range: tuple[float, float],
        quadratic_range: tuple[float, float],
    ) -> tuple[float, SampleSet]:
        """Sample from the graph-restricted Boltzmann machine."""
        self.current_deque_size = 0
        self.iterations_since_last_resampling = 0
        self.deque = None

        resampling_condition = (
            self.current_deque_size < self.max_deque_size
            or self.iterations_since_last_resampling >= self.iterations_before_resampling
        )
        if resampling_condition:
            with torch.no_grad():
                self.sample_set = grbm.sample(
                    sampler,
                    prefactor=prefactor,
                    linear_range=linear_range,
                    quadratic_range=quadratic_range,
                    sample_params=sampler_kwargs,
                    as_tensor=False,
                )
        else:
            num_reads = sampler_kwargs["num_reads"]
            idx = torch.randint(0, self.max_deque_size, size=(num_reads,))
            sampleset_tensor = self.deque[idx]

            self.sample_set = SampleSet.from_samples(
                samples_like=as_samples(sampleset_tensor.numpy()),
                vartype=self.sample_set.vartype,
                energy=self.sample_set.record.energy,
            )

        if resampling_condition:
            sampleset_tensor = grbm.sampleset_to_tensor(self.sample_set)
            if self.deque is None:
                self.deque = sampleset_tensor
            else:
                self.deque = push_to_deque(
                    self.deque,
                    sampleset_tensor[: self.max_deque_size - self.deque.shape[0]],
                    deque_size=self.max_deque_size,
                )
            self.current_deque_size = self.deque.shape[0]
            self.iterations_since_last_resampling = 0
        else:
            self.iterations_since_last_resampling += 1

        return self.sample_set
