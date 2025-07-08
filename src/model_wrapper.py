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


from pathlib import Path
from typing import Optional
import plotly.express as px
from plotly import graph_objects as go
from plotly.subplots import make_subplots
import yaml

from .losses import nll_loss

from .utils.common import get_latent_to_discrete, get_sampler_and_sampler_kwargs
from .utils.persistent_qpu_sampler import PersistentQPUSampleHelper
from .encoder import Encoder
from .decoder import Decoder
# from .encoder import EncoderV2 as Encoder
# from .decoder import DecoderV2 as Decoder

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision.transforms import Compose, Resize, ToTensor
from torchvision.utils import make_grid

from dwave.plugins.torch.autoencoder import DiscreteAutoEncoder
from dwave.plugins.torch.boltzmann_machine import GraphRestrictedBoltzmannMachine

# from dwave.plugins.torch.autoencoder.losses.mmd import RadialBasisFunction, mmd_loss
from .losses import RadialBasisFunction, mmd_loss


def train_dvae(opt_step: int, epoch: int) -> bool:
    """Schedule for training the DVAE.

    Args:
        opt_step: The current optimization step.
        epoch: The current epoch.
    """
    return True


def train_grbm(opt_step: int, epoch: int) -> bool:
    """Schedule for training the GRBM.

    Args:
        opt_step: The current optimization step.
        epoch: The current epoch.
    """
    if epoch < 6 and opt_step % 10 == 0:
        return True
    return False


class TrainingError(Exception):
    """Error when training the model."""


class ModelWrapper:
    """Container class for the discrete VAE w. GRBM model.

    Args:
        n_latents: The number of latent variables in the model.
    """

    def __init__(self, n_latents: Optional[int] = None) -> None:
        self.n_latents: int = n_latents

        self._dvae = None
        self._grbm = None
        self._prefactor = None

        self._device = None
        self.sampler = None
        self.sampler_kwargs = None

        # self.optimizer = None
        self._dvae_optimizer = None
        self._grbm_optimizer = None

        self._dataloader = None

        with open("src/training_parameters.yaml", "r") as f:
            self._params = yaml.safe_load(f)

    def __getattr__(self, name: str):
        if name in self._params:
            return self._params[name]
        return super().__getattribute__(name)

    def save(self, file_path: Optional[str] = None) -> None:
        """Save the model and configs.

        Args:
            file_path: Relative path to the folder where the model should be saved.
        """
        if isinstance(file_path, str):
            file_path = Path(file_path)

        file_path.mkdir(exist_ok=True)

        # Save the model
        torch.save(self._dvae.state_dict(), file_path / "dvae.pth")
        # Save the RBM
        torch.save(self._grbm.state_dict(), file_path / "grbm.pth")
        # Save the prefactor
        torch.save(self._prefactor, file_path / "prefactor.pth")

    def load(self, file_path: str) -> None:
        """Load and reconstruct autoencoder from saved models and configs.

        Args:
            file_path: Relative path to the folder containing the saved model.
        """
        self.setup()
        self._load_dataset(batch_size=self.BATCH_SIZE, dataset_size=self.DATASET_SIZE)

        # currently assuming config and and model have same base name
        self._dvae.load_state_dict(torch.load(file_path / "dvae.pth"))
        self._grbm.load_state_dict(torch.load(file_path / "grbm.pth"))
        self._prefactor = torch.load(
                file_path / "prefactor.pth", weights_only=False
        )

    def setup(self) -> None:
        """Initial setup for the VAE and GRBM."""
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if self.LATENT_TO_DISCRETE in ["heaviside"] and self.N_REPLICAS != 1:
            raise ValueError("heaviside latent-to-discrete can only be used with n_replicas=1")

        dvae = DiscreteAutoEncoder(
            encoder=Encoder(n_latents=self.n_latents),
            decoder=Decoder(n_latents=self.n_latents),
            latent_to_discrete=get_latent_to_discrete(self.LATENT_TO_DISCRETE)
        )

        self._dvae = dvae.to(self._device)

        self.sampler, graph, h_range, j_range, self.sampler_kwargs = get_sampler_and_sampler_kwargs(
            num_reads=self.NUM_READS,
            annealing_time=self.ANNEALING_TIME,
            n_latents=self.n_latents,
            random_seed=self.RANDOM_SEED,
            use_qpu=self.USE_QPU,
        )

        grbm = GraphRestrictedBoltzmannMachine(
            self.n_latents,
            *torch.tensor(list(graph.edges)).mT,
            h_range=h_range,
            j_range=j_range,
        )
        self._grbm = grbm.to(self._device)

        self._dvae_optimizer = torch.optim.Adam(
            self._dvae.parameters(),
            lr=self.AUTOENCODER_INITIAL_LR,
            weight_decay=self.AUTOENCODER_WEIGHT_DECAY,
        )
        self._grbm_optimizer = torch.optim.Adam(
            self._grbm.parameters(),
            lr=self.BM_INITIAL_LR,
            weight_decay=self.BM_WEIGHT_DECAY,
        )

    def _load_dataset(self, batch_size: int, dataset_size: Optional[int] = None) -> None:
        """Load the MNIST dataset and create the dataloader.

        Args:
            batch_size: The batch size to use.
            dataset_size: The number of images to use for training.
                Default (``None``) uses all available images.
        """
        transform = Compose(
            [
                Resize((self.IMAGE_SIZE, self.IMAGE_SIZE)),
                ToTensor(),
                lambda x: torch.round(x),  # Round values to 0 or 1
            ]
        )

        # Load the dataset and create the dataloader
        dataset = MNIST(
            root="data",
            train=True,
            download=True,
            transform=transform,
        )
        if dataset_size:
            dataset = torch.utils.data.random_split(dataset, [dataset_size, len(dataset) - dataset_size])[0]

        self._dataloader = DataLoader(
            dataset, batch_size=batch_size, shuffle=True, drop_last=True
        )

    def train_init(
            self,
            n_epochs: int,
            perturb_grbm: bool = False,
            noise: Optional[float] = None,
        ) -> None:
        """Initialize the model for training.

        Args:
            n_epochs: Number of epochs to train. Used to determine the learning rate schedules.
            perturb_grbm: Whether to perturb the GRBM parameters before training.
            noise: Only used if ``perturb_grbm`` is ``True``.
        """
        # set the random seed for reproducibility
        torch.manual_seed(self.RANDOM_SEED)

        # initialize and store training parameters in a
        # temporary dict to be accessed by the step method
        self._tpar = {}

        self._tpar["persistent_qpu_sample_helper"] = PersistentQPUSampleHelper(
            max_deque_size=self.MAX_DEQUE_SIZE,
            iterations_before_resampling=self.ITERATIONS_BEFORE_RESAMPLING,
        )

        if self._dvae is None or self._grbm is None:
            self.setup()

        if self._dataloader is None:
            self._load_dataset(batch_size=self.BATCH_SIZE, dataset_size=self.DATASET_SIZE)

        n_batches = len(self._dataloader)

        if perturb_grbm:
            self.perturb_grbm_parameters(noise=noise)

        total_opt_steps = n_epochs * n_batches

        self._tpar["dvae_lr_schedule"] = np.geomspace(
            self.AUTOENCODER_INITIAL_LR, self.AUTOENCODER_FINAL_LR, total_opt_steps + 1
        )
        self._tpar["grbm_lr_schedule"] = np.geomspace(
            self.BM_INITIAL_LR, self.BM_FINAL_LR, total_opt_steps + 1
        )

        self._tpar["opt_step"] = 0

        # use for self.LOSS_FUNCTION == "mmd":
        self._tpar["kernel"] = RadialBasisFunction(num_features=7).to(self._device)

        self._tpar["mse_losses"] = []
        self._tpar["dvae_losses"] = []
        self._prefactor = self.INITIAL_PREFACTOR
        self._tpar["last_prefactor"] = self._prefactor
        self._tpar["sample_set"] = None
        self._tpar["alpha"] = 2 / (self.WINDOW_LENGTH + 1)

        self._tpar["init_done"] = True

    def step(self, batch: tuple[torch.Tensor, torch.Tensor], epoch: int) -> torch.Tensor:
        """Train the model on a single batch.

        Args:
            batch: The batch to train on.
            epoch: The current epoch (used to determine training based on schedule).

        Returns:
            torch.Tensor: MSE loss from training step.
        """
        if not self._tpar.get("init_done", True):
            raise TrainingError("Initialization required before training.")

        images, _ = batch
        images = images.to(self._device)
        self._dvae.train()
        self._grbm.train()

        reconstructed_images, spins, _ = self._dvae(images, self.N_REPLICAS)

        # train autoencoder
        if train_dvae(self._tpar["opt_step"], epoch):
            self._dvae_optimizer.zero_grad()
            mse_loss = torch.nn.functional.mse_loss(
                reconstructed_images,
                images.unsqueeze(1).repeat(1, self.N_REPLICAS, 1, 1, 1),
            )
            self._tpar["mse_losses"].append(mse_loss.item())

            dvae_loss = mmd_loss(
                spins=spins,
                kernel=self._tpar["kernel"],
                grbm=self._grbm,
                sampler=self.sampler,
                sampler_kwargs=self.sampler_kwargs,
                prefactor=self._prefactor,
            )

            dvae_loss = mse_loss + dvae_loss

            self._tpar["dvae_losses"].append(dvae_loss.item())

            dvae_loss.backward()
            self._dvae_optimizer.step()

        # train boltzmann machine
        if train_grbm(self._tpar["opt_step"], epoch):
            self._grbm_optimizer.zero_grad()
            self._prefactor, grbm_loss, self._tpar["sample_set"] = nll_loss(
                spins=spins.detach(),
                grbm=self._grbm,
                sampler=self.sampler,
                sampler_kwargs=self.sampler_kwargs,
                prefactor=self._prefactor,
                measure_prefactor=self.MEASURE_PREFACTOR,
                persistent_qpu_sample_helper=self._tpar["persistent_qpu_sample_helper"],
                sample_set=self._tpar["sample_set"],
            )
            grbm_loss.backward()
            self._grbm_optimizer.step()

        self._prefactor = self._tpar["alpha"] * self._prefactor + (1 - self._tpar["alpha"]) * self._tpar["last_prefactor"]
        self._tpar["last_prefactor"] = self._prefactor

        # update learning rate
        for param_group in self._dvae_optimizer.param_groups:
            param_group["lr"] = self._tpar["dvae_lr_schedule"][self._tpar["opt_step"]]
        for param_group in self._grbm_optimizer.param_groups:
            param_group["lr"] = self._tpar["grbm_lr_schedule"][self._tpar["opt_step"]]
        self._tpar["opt_step"] += 1

        return mse_loss


    def perturb_grbm_parameters(self, noise: float) -> None:
        """Perturb the GRBM parameters pre model tuning.

        Args:
            noise: Noise for perturbing the model parameters.

        Returns:
            go.Figure: Plotly figure.
        """
        ...


    def generate_output(self) -> go.Figure:
        """Generate output images from trained model.

        Returns:
            go.Figure: Plotly figure.
        """
        images_per_row = 16
        self._dvae.eval()
        self._grbm.eval()

        with torch.no_grad():
            self._grbm.h.mul_(self._prefactor)
            self._grbm.J.mul_(self._prefactor)
            samples = self._grbm.sample(self.sampler, device=self._device, **self.sampler_kwargs)
            self._grbm.h.mul_(1 / self._prefactor)
            self._grbm.J.mul_(1 / self._prefactor)
        images = (
            self._dvae.decoder(samples.unsqueeze(1))
            .squeeze(1)
            .clip(0.0, 1.0)
            .detach()
            .cpu()
        )
        generation_tensor_for_plot = make_grid(images, nrow=images_per_row)

        fig = px.imshow(generation_tensor_for_plot.permute(1, 2, 0))

        fig.update_xaxes(showticklabels=False)
        fig.update_yaxes(showticklabels=False)
        return fig

    def generate_loss_plot(self, mse_losses, dvae_losses) -> go.Figure:
        """Generate the loss plots for MSE and DVAE loss.

        Args:
            mse_losses: The MSE losses to plot.
            dvae_losses: The DVAE training losses to plot.

        Returns:
            go.Figure: Plotly figure.
        """
        fig = make_subplots(rows=2, cols=1, subplot_titles=("MSE Loss", "Other Loss"))

        fig.add_trace(go.Scatter(x=list(range(len(mse_losses))), y=mse_losses), row=1, col=1)
        fig.add_trace(go.Scatter(x=list(range(len(mse_losses))), y=dvae_losses), row=2, col=1)

        # Update xaxis properties
        fig.update_xaxes(title_text="Batch", row=1, col=1)
        fig.update_yaxes(title_text="Loss", row=1, col=1)

        # Update yaxis properties
        fig.update_xaxes(title_text="Batch", row=2, col=1)
        fig.update_yaxes(title_text="Loss", row=2, col=1)

        return fig

    def generate_reconstucted_samples(self) -> go.Figure:
        """Generate reconstructed images from training data.

        Returns:
            go.Figure: Plotly figure.
        """
        images_per_row = 16
        # Now we use the trained autoencoder both to generate new samples as well as to
        # show the reconstruction of the input samples.
        batch = next(iter(self._dataloader))[0]
        self._dvae.eval()
        self._grbm.eval()
        reconstructed_batch, _, _ = self._dvae(batch.to(self._device))
        reconstruction_tensor_for_plot = make_grid(
            torch.cat(
                (
                    batch.cpu(),
                    torch.ones((images_per_row, 1, self.IMAGE_SIZE, self.IMAGE_SIZE)),
                    reconstructed_batch.clip(0.0, 1.0).squeeze(1).cpu(),
                ),
                dim=0,
            ),
            nrow=images_per_row,
        )
        fig = px.imshow(reconstruction_tensor_for_plot.permute(1, 2, 0))

        fig.update_xaxes(showticklabels=False)
        fig.update_yaxes(showticklabels=False)
        return fig

    def generate_training_data(self) -> go.Figure:
        """Generate original images used for training."""
        # TODO: return unalterted training images from dataset
