import torch
from dimod import BinaryQuadraticModel, Sampler, SampleSet, as_samples
from dwave.system.temperatures import maximum_pseudolikelihood_temperature

from dwave.plugins.torch.boltzmann_machine import GraphRestrictedBoltzmannMachine
from dwave.plugins.torch.utils import sample_to_tensor, spread


def push_to_deque(deque, x, deque_size=None, dim=0):
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
                (
                    input_size - deque_size
                    if input_size < deque_size
                    else deque.shape[dim]
                ),
                None,
            ),
        )
        + (slice(None),) * dims_right
    )
    input_slicing = (
        (slice(None),) * dim + (slice(-deque_size, None),) + (slice(None),) * dims_right
    )
    deque = torch.cat((deque[deque_slicing], x[input_slicing]), dim=dim)
    return deque


def sample_from_grbm(
    grbm: GraphRestrictedBoltzmannMachine,
    sampler: Sampler,
    sampler_kwargs: dict,
    prefactor: float,
) -> SampleSet:
    with torch.no_grad():
        grbm.h.mul_(prefactor)
        grbm.J.mul_(prefactor)
        h, J = grbm.ising
        sample_set = sampler.sample_ising(h, J, **sampler_kwargs)
        grbm.h.mul_(1 / prefactor)
        grbm.J.mul_(1 / prefactor)
    return sample_set


def find_prefactor(
    previous_prefactor: float,
    grbm: GraphRestrictedBoltzmannMachine,
    sampler: Sampler,
    sampler_kwargs: dict,
    atol: float = 0.05,
    measure_prefactor: bool = True,
    sample_set: SampleSet | None = None,
) -> tuple[float, SampleSet]:
    if sample_set is None:
        sample_set = sample_from_grbm(grbm, sampler, sampler_kwargs, previous_prefactor)
    h, J = grbm.ising

    if measure_prefactor:
        bqm = BinaryQuadraticModel.from_ising(h, J)
        temperature, _ = maximum_pseudolikelihood_temperature(
            bqm, sample_set, T_guess=1.0
        )
        beta = 1 / temperature
    else:
        temperature = 1
        beta = 1

    if beta < 1 - atol or beta > 1 + atol:
        return find_prefactor(
            temperature * previous_prefactor, grbm, sampler, sampler_kwargs, atol
        )
    else:
        return temperature * previous_prefactor, sample_set


class PersistentQPUSampleHelper:
    def __init__(self, max_deque_size: int, iterations_before_resampling: int):
        self.current_deque_size = 0
        self.max_deque_size = max_deque_size
        self.iterations_before_resampling = iterations_before_resampling
        self.iterations_since_last_resampling = 0
        self.deque = None

    def find_prefactor(
        self,
        previous_prefactor: float,
        grbm: GraphRestrictedBoltzmannMachine,
        sampler: Sampler,
        sampler_kwargs: dict,
        atol: float = 0.05,
        measure_prefactor: bool = True,
        resample: bool = False,
        reset_deque: bool = False,
    ):
        if reset_deque:
            self.current_deque_size = 0
            self.iterations_since_last_resampling = 0
            self.deque = None
            return self.find_prefactor(
                previous_prefactor,
                grbm,
                sampler,
                sampler_kwargs,
                atol,
                measure_prefactor,
                resample,
                reset_deque=False,
            )
        resampling_condition = (
            self.current_deque_size < self.max_deque_size
            or self.iterations_since_last_resampling
            >= self.iterations_before_resampling
            or resample
        )
        num_reads = sampler_kwargs["num_reads"]
        if resampling_condition:
            sample_set = None
        else:
            assert isinstance(self.deque, torch.Tensor)
            idx = torch.randint(0, self.max_deque_size, size=(num_reads,))
            tensor = self.deque[idx]
            sample_set = SampleSet.from_samples(
                samples_like=as_samples(tensor.numpy()),
                vartype=self.sample_set_vartype,
                energy=self.sample_set_energy,
            )

        prefactor, sample_set = find_prefactor(
            previous_prefactor=previous_prefactor,
            grbm=grbm,
            sampler=sampler,
            sampler_kwargs=sampler_kwargs,
            atol=atol,
            measure_prefactor=measure_prefactor,
            sample_set=sample_set,
        )
        self.sample_set_vartype = sample_set.vartype
        self.sample_set_energy = sample_set.record.energy
        if resampling_condition:
            tensor = sample_to_tensor(spread(sample_set))
            if self.deque is None:
                self.deque = tensor
            else:
                self.deque = push_to_deque(
                    self.deque,
                    tensor[: self.max_deque_size - self.deque.shape[0]],
                    deque_size=self.max_deque_size,
                )
            self.current_deque_size = self.deque.shape[0]
            self.iterations_since_last_resampling = 0
        else:
            self.iterations_since_last_resampling += 1
        return prefactor, sample_set
