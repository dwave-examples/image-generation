from dataclasses import dataclass
from typing import Literal, Optional

from matplotlib import pyplot as plt

from src.encoder import Encoder
from src.decoder import Decoder


@dataclass
class AutoEncoderConfig:
    n_latents: int = None


class DiscreteVAE:
    """Container class for the discrete VAE w. GRBM model."""

    def __init__(self, n_latents: Optional[int] = None) -> None:
        self.n_latents: int = n_latents
        self._dvae = None
        self._grbm = None

    def load(self, file_name: str) -> None:
        """Load and reconstruct autoencoder from saved models and configs."""
        # currently assuming config and and model have same base name
        ...
        self._dvae = ...
        self._grbm = ...

    def setup(self) -> None:
        """Initial setup for the VAE and GRBM."""
        ...
        self._dvae = ...
        self._grbm = ...

    def train(self, perturb_grbm: bool = False, noise: Optional[float] = None) -> None:
        """Train the model."""
        if self._dvae is None or self._grbm is None:
            self.setup()

        if perturb_grbm:
            self.perturb_grbm_parameters(noise=noise)
        ...

    def save(file_name: Optional[str] = None) -> None:
        """Save the model and configs."""
        ...

    def perturb_grbm_parameters(self, noise: float) -> None:
        """Perturb the GRBM parameters."""
        ...


    def generate_output(self) -> plt.axes:
        """Generate output images."""
        ...



def run_training(n_latents: int, file_name: Optional[str] = None) -> None:
    """Creates a new VAE, trains it and saves the model and config."""
    dvae = DiscreteVAE(n_latents=n_latents)
    dvae.train(perturb_grbm=False)
    dvae.save(file_name=file_name)


def run_generate(file_name: str, tune_parameters: bool = True, noise: Optional[float] = None):
    """Loads an existing VAE, optionally tunes it, and saves/returns an image."""
    # load autoencoder model and config
    dvae = DiscreteVAE()
    dvae.load(file_name=file_name)

    if tune_parameters:
        dvae.train(perturb_grbm=True, noise=noise)

    ...
    # save or return figure here
