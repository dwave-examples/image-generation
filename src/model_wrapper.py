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

import numpy as np
import plotly.express as px
import torch
import yaml
from dwave.plugins.torch.models import (
    DiscreteVariationalAutoencoder,
    GraphRestrictedBoltzmannMachine,
)
from einops import rearrange
from plotly import graph_objects as go
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision.transforms import Compose, Resize, ToTensor
from torchvision.utils import make_grid

from demo_configs import LOWER_THRESHOLD, UPPER_THRESHOLD

from .decoder import Decoder
from .encoder import Encoder
from .losses import RadialBasisFunction, mmd_loss, nll_loss
from .utils.common import get_latent_to_discrete, get_sampler_and_sampler_kwargs
from .utils.persistent_qpu_sampler import PersistentQPUSampleHelper


def train_dvae(opt_step: int, epoch: int) -> bool:
    """Schedule for training the DVAE.

    Args:
        opt_step: The current optimization step.
        epoch: The current epoch.
    """
    ###TODO: Remove? this is not useful in its current state
    return True


def train_grbm(opt_step: int, epoch: int) -> bool:
    """Schedule for training the GRBM.

    Args:
        opt_step: The current optimization step.
        epoch: The current epoch.
    """

    return epoch < 6 and opt_step % 10 == 0


def get_dataset(image_size: int, batch_size: int, dataset_size: Optional[int] = None) -> DataLoader:
    transform = Compose(
        [
            Resize((image_size, image_size)),
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
        dataset = torch.utils.data.random_split(
            dataset, [dataset_size, len(dataset) - dataset_size]
        )[0]

    return DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)


def display_dataset(dataset: DataLoader, num_rows: int) -> go.Figure:
    batch = next(iter(dataset))[0]
    reconstruction_tensor_for_plot = make_grid(batch.cpu(), nrow=num_rows)
    fig = px.imshow(reconstruction_tensor_for_plot.permute(1, 2, 0))

    fig.update_xaxes(showticklabels=False)
    fig.update_yaxes(showticklabels=False)
    fig.update_layout(
        margin={"t": 0, "l": 0, "b": 0, "r": 0},
        paper_bgcolor="black",
        plot_bgcolor="black",
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
    )
    return fig


class TrainingError(Exception):
    """Error when training the model."""


class ModelWrapper:
    """Container class for the discrete VAE w. GRBM model.

    Args:
        n_latents: The number of latent variables in the model.
    """

    def __init__(
        self, qpu: str, n_latents: Optional[int] = None, training_parameter_file: str = None
    ) -> None:
        self.qpu: str = qpu
        self.n_latents: int = n_latents

        self._dvae = None
        self._grbm = None

        self._device = None
        self.sampler = None
        self.sampler_kwargs = None

        self._dvae_optimizer = None
        self._grbm_optimizer = None

        self._dataloader = None

        self.losses = {"mse_losses": [], "dvae_losses": []}

        if not training_parameter_file:
            training_parameter_file = "src/training_parameters.yaml"

        with open(training_parameter_file, "r") as f:
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

    def setup(self) -> None:
        """Initial setup for the VAE and GRBM."""
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if self.LATENT_TO_DISCRETE in ["heaviside"] and self.N_REPLICAS != 1:
            raise ValueError("heaviside latent-to-discrete can only be used with n_replicas=1")

        dvae = DiscreteVariationalAutoencoder(
            encoder=Encoder(n_latents=self.n_latents),
            decoder=Decoder(n_latents=self.n_latents),
            latent_to_discrete=get_latent_to_discrete(self.LATENT_TO_DISCRETE),
        )

        self._dvae = dvae.to(self._device)

        self.sampler, self.sampler_kwargs, graph, self.linear_range, self.quadratic_range = (
            get_sampler_and_sampler_kwargs(
                num_reads=self.NUM_READS,
                annealing_time=self.ANNEALING_TIME,
                n_latents=self.n_latents,
                random_seed=self.RANDOM_SEED,
                qpu=self.qpu,
            )
        )

        grbm = GraphRestrictedBoltzmannMachine(
            graph.nodes,
            graph.edges,
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
        self._dataloader = get_dataset(self.IMAGE_SIZE, batch_size, dataset_size)

    def train_init(
        self,
        n_epochs: int,
    ) -> None:
        """Initialize the model for training.

        Args:
            n_epochs: Number of epochs to train. Used to determine the learning rate schedules.
        """
        self.losses["mse_losses"].clear()
        self.losses["dvae_losses"].clear()

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

        self._tpar["sample_set"] = None

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

        _, spins, reconstructed_images = self._dvae(images, self.N_REPLICAS)

        # train autoencoder
        if train_dvae(self._tpar["opt_step"], epoch):
            self._dvae_optimizer.zero_grad()
            mse_loss = torch.nn.functional.mse_loss(
                reconstructed_images,
                images.unsqueeze(1).repeat(1, self.N_REPLICAS, 1, 1, 1),
            )
            self.losses["mse_losses"].append(mse_loss.item())

            _mmd_loss = mmd_loss(
                spins=spins,
                kernel=self._tpar["kernel"],
                grbm=self._grbm,
                sampler=self.sampler,
                sampler_kwargs=self.sampler_kwargs,
                linear_range=self.linear_range,
                quadratic_range=self.quadratic_range,
                prefactor=self.PREFACTOR,
            )

            dvae_loss = mse_loss + _mmd_loss

            self.losses["dvae_losses"].append(dvae_loss.item())

            dvae_loss.backward()
            self._dvae_optimizer.step()

        # train boltzmann machine
        if train_grbm(self._tpar["opt_step"], epoch):
            self._grbm_optimizer.zero_grad()
            grbm_loss, self._tpar["sample_set"] = nll_loss(
                spins=spins.detach(),
                grbm=self._grbm,
                sampler=self.sampler,
                sampler_kwargs=self.sampler_kwargs,
                linear_range=self.linear_range,
                quadratic_range=self.quadratic_range,
                prefactor=self.PREFACTOR,
                persistent_qpu_sample_helper=self._tpar["persistent_qpu_sample_helper"],
                sample_set=self._tpar["sample_set"],
            )
            grbm_loss.backward()
            self._grbm_optimizer.step()

        # update learning rate
        for param_group in self._dvae_optimizer.param_groups:
            param_group["lr"] = self._tpar["dvae_lr_schedule"][self._tpar["opt_step"]]
        for param_group in self._grbm_optimizer.param_groups:
            param_group["lr"] = self._tpar["grbm_lr_schedule"][self._tpar["opt_step"]]
        self._tpar["opt_step"] += 1

        return mse_loss

    def generate_output(self, sharpen: bool = False, save_to_file: str = "") -> go.Figure:
        """Generate output images from trained model.

        Args:
            Whether to sharpen the output images by binarization.

        Returns:
            go.Figure: Plotly figure.
        """
        images_per_row = 16
        self._dvae.eval()
        self._grbm.eval()

        with torch.no_grad():
            samples = self._grbm.sample(
                self.sampler,
                prefactor=self.PREFACTOR,
                device=self._device,
                linear_range=self.linear_range,
                quadratic_range=self.quadratic_range,
                sample_params=self.sampler_kwargs,
            )

        images = self._dvae.decoder(samples.unsqueeze(1)).squeeze(1).clip(0.0, 1.0).detach().cpu()
        if sharpen:
            over = (images - UPPER_THRESHOLD).heaviside(torch.tensor([0.0]))
            under = (images - LOWER_THRESHOLD).heaviside(torch.tensor([0.0]))
            images = (over + abs(over - 1) * images) * under

        generation_tensor_for_plot = make_grid(images, nrow=images_per_row)

        fig = px.imshow(generation_tensor_for_plot.permute(1, 2, 0))

        fig.update_xaxes(showticklabels=False)
        fig.update_yaxes(showticklabels=False)
        fig.update_layout(margin={"t": 0,"l": 0,"b": 0,"r": 0})

        if save_to_file:
            with open(save_to_file, "w") as f:
                f.write(fig.to_json())

        return fig

    def generate_loss_plot(self, save_to_file_mse: str="", save_to_file_total: str="") -> tuple[go.Figure, go.Figure]:
        """Generate the loss plots for MSE and DVAE loss.

        Returns:
            go.Figure: The Mean Squared Error losses plot.
            go.Figure: The total losses plot.
        """
        mse_losses = self.losses["mse_losses"]
        dvae_losses = self.losses["dvae_losses"]

        fig_mse = go.Figure()
        fig_total = go.Figure()

        fig_mse.add_trace(go.Scatter(x=list(range(len(mse_losses))), y=mse_losses))
        fig_total.add_trace(go.Scatter(x=list(range(len(mse_losses))), y=dvae_losses))

        # Update xaxis properties
        fig_mse.update_xaxes(title_text="Batch")
        fig_mse.update_yaxes(title_text="Loss")

        # Update yaxis properties
        fig_total.update_xaxes(title_text="Batch")
        fig_total.update_yaxes(title_text="Loss")

        fig_mse.update_layout(margin={"t": 0, "l": 0, "b": 0, "r": 0})
        fig_total.update_layout(margin={"t": 0, "l": 0, "b": 0, "r": 0})

        if save_to_file_mse:
            with open(save_to_file_mse, "w") as f:
                f.write(fig_mse.to_json())

        if save_to_file_total:
            with open(save_to_file_total, "w") as f:
                f.write(fig_total.to_json())

        return fig_mse, fig_total

    def generate_reconstucted_samples(self, sharpen: bool = False, save_to_file: str="") -> go.Figure:
        """Generate reconstructed images from training data.

        Args:
            Whether to sharpen the output images by binarization.

        Returns:
            go.Figure: A figure showing the comparison between original and reconstructed digits.
        """
        images_per_row = 16
        # Now we use the trained autoencoder both to generate new samples as well as to
        # show the reconstruction of the input samples.
        batch = next(iter(self._dataloader))[0]
        self._dvae.eval()
        self._grbm.eval()

        _, _, reconstructed_batch = self._dvae(batch.to(self._device))
        reconstructed_batch[:, :, :, :, -1] = 1.0
        images = make_grid(
            rearrange(
                [batch.cpu(), reconstructed_batch.clip(0.0, 1.0).squeeze(1).cpu()],
                "i b c h w -> (b i) c h w",
            ),
            nrow=images_per_row,
            padding=0,
        )

        if sharpen:
            over = (images - UPPER_THRESHOLD).heaviside(torch.tensor([0.0]))
            under = (images - LOWER_THRESHOLD).heaviside(torch.tensor([0.0]))
            images = (over + abs(over - 1) * images) * under

        fig = px.imshow(images.permute(1, 2, 0))

        fig.update_xaxes(showticklabels=False)
        fig.update_yaxes(showticklabels=False)
        fig.update_layout(margin={"t": 0, "l": 0, "b": 0, "r": 0})

        if save_to_file:
            with open(save_to_file, "w") as f:
                f.write(fig.to_json())

        return fig
