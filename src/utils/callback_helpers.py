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

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Optional

from dwave.plugins.torch.models import DiscreteVariationalAutoencoder
from plotly import graph_objects as go

from demo_configs import SHARPEN_OUTPUT

MODEL_PATH = Path("models")
JSON_FILE_DIR = "generated_json"
PROBLEM_DETAILS_PATH = f"{JSON_FILE_DIR}/problem_details.json"
IMAGE_GEN_FILE_PREFIX = "generated_epoch_"
IMAGE_RECON_FILE_PREFIX = "reconstructed_epoch_"
LOSS_PREFIX = "loss_"


def create_model_files(
    model: DiscreteVariationalAutoencoder,
    file_name: str,
    qpu: str,
    n_latents: int,
    n_epochs: int,
    loss_data: dict,
):
    """Creates model files, losses file, and parameters file.

    Args:
        model: The DVAE model.
        file_name: The directory name to save all the files to.
        qpu: The QPU associated with the model.
        n_latents: The number of latents.
        n_epochs: The number of epochs.
        loss_data: The loss data to save.
    """
    model.save(file_path=MODEL_PATH / file_name)

    with open(MODEL_PATH / file_name / "parameters.json", "w") as f:
        json.dump(
            {
                "n_latents": n_latents,
                "n_epochs": n_epochs,
                "prefactor": model.PREFACTOR,
                "qpu": qpu,
                "num_read": model.NUM_READS,
                "loss_function": model.LOSS_FUNCTION,
                "image_size": model.IMAGE_SIZE,
                "batch_size": model.BATCH_SIZE,
                "dateset_size": model.DATASET_SIZE,
            },
            f,
        )

    with open(MODEL_PATH / file_name / "losses.json", "w") as f:
        json.dump(loss_data, f)


def execute_training(
    set_progress,
    model: DiscreteVariationalAutoencoder,
    n_epochs: int,
    qpu: str,
    n_latents: int,
    loss_data: Optional[list] = None,
) -> tuple[go.Figure, go.Figure, go.Figure, go.Figure]:
    """Orchestrates training or tuning of model.

    Args:
        model: The DVAE model.
        n_epochs: The number of epochs to run training for.
        qpu: The selected QPU.
        n_latents: The size of the latent space for the training.
        loss_data: Old loss data from previous training.

    Returns:
        fig_output: The generated image output.
        fig_reconstructed: The image comparing the reconstructed image to the original.
        fig_mse_loss: The graph showing the MSE Loss.
        fig_total_loss: The graph showing the total Loss (MMD + MSE).
    """
    for epoch in range(n_epochs):
        start_time = time.perf_counter()
        print(f"Starting epoch {epoch + 1}/{n_epochs}")

        total = len(model._dataloader)
        for i, batch in enumerate(model._dataloader):
            set_progress((str(total * epoch + i), str(total * n_epochs)))
            mse_loss = model.step(batch, epoch)

        learning_rate_dvae = model._tpar["dvae_lr_schedule"][model._tpar["opt_step"]]
        learning_rate_grbm = model._tpar["grbm_lr_schedule"][model._tpar["opt_step"]]
        print(
            f"Epoch {epoch + 1}/{n_epochs} - MSE Loss: {mse_loss.item():.4f} - "
            f"Learning rate DVAE: {learning_rate_dvae:.3E} "
            f"Learning rate GRBM: {learning_rate_grbm:.3E} "
            f"Time: {(time.perf_counter() - start_time)/60:.2f} mins. "
        )
        with open(PROBLEM_DETAILS_PATH, "w") as f:
            json.dump(
                {
                    "QPU": qpu,
                    "Epoch": f"{epoch + 1}/{n_epochs}",
                    "Batch Size": model.BATCH_SIZE,
                    "Latents": n_latents,
                    "Learning rate DVAE": f"{learning_rate_dvae:.3E}",
                    "Learning rate GRBM": f"{learning_rate_grbm:.3E}",
                    "Mean Squared Error Loss": f"{mse_loss.item():.4f}",
                },
                f,
            )

        fig_output = model.generate_output(
            sharpen=SHARPEN_OUTPUT,
            save_to_file=f"{JSON_FILE_DIR}/{IMAGE_GEN_FILE_PREFIX}{epoch+1}.json",
        )
        fig_reconstructed = model.generate_reconstucted_samples(
            sharpen=SHARPEN_OUTPUT,
            save_to_file=f"{JSON_FILE_DIR}/{IMAGE_RECON_FILE_PREFIX}{epoch+1}.json",
        )
        fig_mse_loss, fig_dvae_loss = model.generate_loss_plot(
            save_to_file_mse=f"{JSON_FILE_DIR}/{LOSS_PREFIX}mse_{epoch+1}.json",
            save_to_file_total=f"{JSON_FILE_DIR}/{LOSS_PREFIX}total_{epoch+1}.json",
            old_loss_data=loss_data,
        )

    return fig_output, fig_reconstructed, fig_mse_loss, fig_dvae_loss
