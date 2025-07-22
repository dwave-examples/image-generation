# Copyright 2024 D-Wave
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
import math
import os
import time
import yaml
from pathlib import Path

import dash
from dash import MATCH
from dash.dependencies import Input, Output, State
from dwave.cloud import Client
from plotly import graph_objects as go

from demo_interface import generate_options
from src.model_wrapper import ModelWrapper

MODEL_PATH = Path("models")


@dash.callback(
    Output({"type": "to-collapse-class", "index": MATCH}, "className"),
    inputs=[
        Input({"type": "collapse-trigger", "index": MATCH}, "n_clicks"),
        State({"type": "to-collapse-class", "index": MATCH}, "className"),
    ],
    prevent_initial_call=True,
)
def toggle_left_column(collapse_trigger: int, to_collapse_class: str) -> str:
    """Toggles a 'collapsed' class that hides and shows some aspect of the UI.

    Args:
        collapse_trigger (int): The (total) number of times a collapse button has been clicked.
        to_collapse_class (str): Current class name of the thing to collapse, 'collapsed' if not
            visible, empty string if visible.

    Returns:
        str: The new class name of the thing to collapse.
    """

    classes = to_collapse_class.split(" ") if to_collapse_class else []
    if "collapsed" in classes:
        classes.remove("collapsed")
        return " ".join(classes)
    return to_collapse_class + " collapsed" if to_collapse_class else "collapsed"


@dash.callback(
    Output("popup", "className", allow_duplicate=True),
    Input("popup-toggle", "n_clicks"),
    prevent_initial_call=True,
)
def toggle_popup(popup_toggle: list[int]) -> str:
    """Hide popup when close button is clicked.

    Args:
        popup_toggle: The close button for the popup toggle.

    Returns:
        popup-classname: The class name to hide the popup.
    """
    return "display-none"


@dash.callback(
    Output("popup", "className"),
    Output("generate-button", "disabled"),
    Input("model-file-name", "value"),
)
def check_qpu_availability(model_file_name: str) -> tuple[str, bool]:
    """Checks whether user has access to QPU associated with model when model changes.

    Args:
        model: The currently selected model

    Returns:
        popup-classname: The class name to hide the popup.
        generate-button-disabled: Whether to disable or enable the Generate button.
    """
    with open(MODEL_PATH / model_file_name / "parameters.json") as file:
        model_data = json.load(file)

    if model_data["qpu"]:
        try:
            client = Client.from_config(client="qpu")
            SOLVERS = [qpu.name for qpu in client.get_solvers()]

            if not len(SOLVERS) or model_data["qpu"] not in SOLVERS:
                raise Exception

        except Exception:
            return "", True

    return "display-none", False


@dash.callback(
    Output("tune-parameter-settings", "className"),
    Input("tune-params", "value"),
)
def toggle_tuning_params(tune_params: list[int]) -> str:
    """Show/hide tune parameter settings when Tune Parameters box is toggled.

    Args:
        tune_params: The value of the Tune Parameters checkbox as a list.

    Returns:
        tune-parameter-settings-classname: The class name to show/hide the tune parameter settings.
    """
    return "" if len(tune_params) else "display-none"


@dash.callback(
    Output("model-file-name", "options"),
    Output("model-file-name", "value"),
    Output("batch-size", "data"),
    Input("last-trained-model", "data"),
)
def initialize_training_model(last_trained_model: str) -> tuple[list[str], str, int]:
    """Initializes the Trained Models dropdown options based on model files available and sets
    the batch size data store.

    Args:
        last_trained_model: The most recently trained model directiory name.

    Returns:
        model-file-name-options: The options for the Trained Model dropdown selection.
        model-file-name-value: The value of the dropdown.
        batch-size-data: The batch size that was read from the model param file.
    """
    models = []
    project_directory = os.path.dirname(os.path.realpath(__file__))

    models_dir = os.path.join(project_directory, "models")
    directories = os.fsencode(models_dir)

    for dir in os.listdir(directories):
        directory = os.fsdecode(dir)
        models.append(directory)

    if not len(models):
        models = generate_options(["No Models Found (please train and save a model)"])

    with open("src/training_parameters.yaml", "r") as f:
        parameters = yaml.safe_load(f)

    return models, last_trained_model if last_trained_model else models[0], parameters["BATCH_SIZE"]


@dash.callback(
    Output({"type": "progress-caption-epoch", "index": MATCH}, "children"),
    Output({"type": "progress-caption-batch", "index": MATCH}, "children"),
    Output({"type": "progress-wrapper", "index": MATCH}, "className"),
    inputs=[
        Input({"type": "progress", "index": MATCH}, "value"),
        State({"type": "progress", "index": MATCH}, "max"),
        State("n-epochs-tune", "value"),
        State("batch-size", "data"),
    ],
    prevent_initial_call=True,
)
def update_progress(
    progress_value: str,
    progress_max: str,
    n_epochs: int,
    batch_size: int
) -> tuple[str, str, str]:
    """Updates progress bar with epochs and batches completed.

    Args:
        progress_value: The current value of the progress bar.
        progress_max: The maximum value of the progress bar ie progress_value/progress_max.
        n_epochs: The number of epochs to complete.
        batch_size: The number of items in a batch.

    Returns:
        progress-caption-epoch: The caption of the progress bar that tracks the completed epochs.
        progress-caption-batch: The caption of the progress bar that tracks the completed batches.
        progress-wrapper-className: The classname of the progress wrapper.
    """
    progress_value = int(progress_value) if progress_value else 0
    progress_max = int(progress_max) if progress_max else 0

    curr_epoch = math.floor(progress_value/(n_epochs*batch_size))

    return (
        f"Epochs Completed: {curr_epoch}/{n_epochs}",
        f"Batch: {progress_value%(n_epochs*batch_size)}/{math.floor(progress_max/n_epochs)}",
        "",
    )


@dash.callback(
    Output("fig-output", "figure", allow_duplicate=True),
    Output("fig-loss", "figure", allow_duplicate=True),
    Output("fig-reconstructed", "figure", allow_duplicate=True),
    Output("last-trained-model", "data"),
    background=True,
    inputs=[
        Input("train-button", "n_clicks"),
        State("qpu-setting", "value"),
        State("n-latents", "value"),
        State("n-epochs", "value"),
        State("file-name", "value"),
    ],
    running=[
        (Output("cancel-training-button", "className"), "", "display-none"),
        (Output("train-button", "className"), "display-none", ""),
        (Output("results-tab", "disabled"), True, False),  # Disables results tab while running.
        (Output("loss-tab", "disabled"), True, False),  # Disables loss tab while running.
        (Output("results-tab", "label"), "Loading...", "Generated Images"),
        (Output("loss-tab", "label"), "Loading...", "Loss Graphs"),
        (Output("tabs", "value"), "input-tab", "input-tab"),  # Switch to input tab while running.
    ],
    cancel=[Input("cancel-training-button", "n_clicks")],
    progress=[
        Output({"type": "progress", "index": 0}, "value"),
        Output({"type": "progress", "index": 0}, "max"),
    ],
    prevent_initial_call=True,
)
def train(
    set_progress, train_click: int, qpu: str, n_latents: int, n_epochs: int, file_name: str
) -> tuple[go.Figure, go.Figure, go.Figure, str]:
    """Runs training and updates UI accordingly.

    This function is called when the ``Train`` button is clicked. It takes in all form values and
    runs the training, updates the run/cancel buttons, deactivates (and reactivates) the results
    tab, and updates all relevant HTML components.

    Args:
        train_click: The (total) number of times the train button has been clicked.
        qpu: The selected QPU.
        n_latents: The value of the latents setting.
        n_epochs: The value of th epochs setting
        file_name: The file name to save to.

    Returns:
        A tuple containing all outputs to be used when updating the HTML
        template (in ``demo_interface.py``). These are:

            fig-output: The generated image output.
            fig-loss: The graphs showing the MSE Loss and Other Loss.
            fig-reconstructed: The image comparing the reconstructed image to the original.
            last-trained-model: The directory name of the model trained by this run.
    """
    dvae = ModelWrapper(qpu=qpu, n_latents=n_latents)

    dvae.train_init(n_epochs)

    for epoch in range(n_epochs):
        start_time = time.perf_counter()
        print(f"Starting epoch {epoch + 1}/{n_epochs}")

        total = len(dvae._dataloader)
        for i, batch in enumerate(dvae._dataloader):
            set_progress((str(total * epoch + i), str(total * n_epochs)))
            mse_loss = dvae.step(batch, epoch)

        print(
            f"Epoch {epoch + 1}/{n_epochs} - MSE Loss: {mse_loss.item():.4f} - "
            f'Learning rate DVAE: {dvae._tpar["dvae_lr_schedule"][dvae._tpar["opt_step"]]:.3E} '
            f'Learning rate GRBM: {dvae._tpar["grbm_lr_schedule"][dvae._tpar["opt_step"]]:.3E} '
            f"Time: {(time.perf_counter() - start_time)/60:.2f} mins. "
        )

    dvae.save(file_path=MODEL_PATH / file_name)

    with open(MODEL_PATH / file_name / "parameters.json", "w") as f:
        json.dump(
            {
                "n_latents": n_latents,
                "qpu": qpu,
                "num_read": dvae.NUM_READS,
                "loss_function": dvae.LOSS_FUNCTION,
                "image_size": dvae.IMAGE_SIZE,
                "batch_size": dvae.BATCH_SIZE,
                "dateset_size": dvae.DATASET_SIZE,
            },
            f,
        )

    with open(MODEL_PATH / file_name / "losses.json", "w") as f:
        json.dump(
            {
                "mse_losses": dvae._tpar["mse_losses"],
                "dvae_losses": dvae._tpar["dvae_losses"],
            },
            f,
        )

    mse_losses, dvae_losses = dvae._tpar["mse_losses"], dvae._tpar["dvae_losses"]

    fig_output = dvae.generate_output()
    if mse_losses and dvae_losses:
        fig_loss = dvae.generate_loss_plot(mse_losses, dvae_losses)

    fig_reconstructed = dvae.generate_reconstucted_samples()

    return fig_output, fig_loss, fig_reconstructed, file_name


@dash.callback(
    Output("fig-output", "figure"),
    Output("fig-loss", "figure"),
    Output("fig-reconstructed", "figure"),
    Output("popup", "className", allow_duplicate=True),
    background=True,
    inputs=[
        Input("generate-button", "n_clicks"),
        State("model-file-name", "value"),
        State("tune-params", "value"),
        State("n-epochs-tune", "value"),
    ],
    running=[
        (Output("cancel-generation-button", "className"), "", "display-none"),
        (Output("generate-button", "className"), "display-none", ""),
        (Output("results-tab", "disabled"), True, False),  # Disables results tab while running.
        (Output("loss-tab", "disabled"), True, False),  # Disables loss tab while running.
        (Output("results-tab", "label"), "Loading...", "Generated Images"),
        (Output("loss-tab", "label"), "Loading...", "Loss Graphs"),
        (Output("tabs", "value"), "input-tab", "input-tab"),  # Switch to input tab while running.
    ],
    progress=[
        Output({"type": "progress", "index": 1}, "value"),
        Output({"type": "progress", "index": 1}, "max"),
    ],
    cancel=[Input("cancel-generation-button", "n_clicks")],
    prevent_initial_call=True,
)
def generate(
    set_progress,
    generate_click: int,
    model_file_name: str,
    tune_parameters: list,
    n_epochs: int,
) -> tuple[go.Figure, go.Figure, go.Figure]:
    """Runs generation and updates UI accordingly.

    This function is called when the ``Generate`` button is clicked. It takes in all form values and
    runs the generation, updates the run/cancel buttons, deactivates (and reactivates) the results
    tab, and updates all relevant HTML components.

    Args:
        generate_click: The (total) number of times the generate button has been clicked.
        model_file_name: The currently selected model directory name.
        tune_parameters: Whether to tune the parameters while generating.
        n_epochs: The number of epochs for the parameter tuning.

    Returns:
        A tuple containing all outputs to be used when updating the HTML
        template (in ``demo_interface.py``). These are:

            fig-output: The generated image output.
            fig-loss: The graphs showing the MSE Loss and Other Loss.
            fig-reconstructed: The image comparing the reconstructed image to the original.
    """
    # load autoencoder model and config
    with open(MODEL_PATH / model_file_name / "parameters.json") as file:
        model_data = json.load(file)
    with open(MODEL_PATH / model_file_name / "losses.json") as file:
        loss_data = json.load(file)

    if model_data["qpu"]:
        try:
            client = Client.from_config(client="qpu")
            SOLVERS = [qpu.name for qpu in client.get_solvers()]

            if not len(SOLVERS) or model_data["qpu"] not in SOLVERS:
                raise Exception

        except Exception:
            return dash.no_update, dash.no_update, dash.no_update, ""

    dvae = ModelWrapper(qpu=model_data["qpu"], n_latents=model_data["n_latents"])
    dvae.load(file_path=MODEL_PATH / model_file_name)

    if tune_parameters:
        dvae.train_init(n_epochs)

        for epoch in range(n_epochs):
            start_time = time.perf_counter()
            print(f"Starting epoch {epoch + 1}/{n_epochs}")

            total = len(dvae._dataloader)
            for i, batch in enumerate(dvae._dataloader):
                print(f"{i}/{total}")
                set_progress((str(total * epoch + i), str(total * n_epochs)))
                mse_loss = dvae.step(batch, epoch)

            print(
                f"Epoch {epoch + 1}/{n_epochs} - MSE Loss: {mse_loss.item():.4f} - "
                f'Learning rate DVAE: {dvae._tpar["dvae_lr_schedule"][dvae._tpar["opt_step"]]:.3E} '
                f'Learning rate GRBM: {dvae._tpar["grbm_lr_schedule"][dvae._tpar["opt_step"]]:.3E} '
                f"Time: {(time.perf_counter() - start_time)/60:.2f} mins. "
            )

        model_file_name += f"_tuned_{n_epochs}"

        loss_data["mse_losses"] += dvae._tpar["mse_losses"]
        loss_data["dvae_losses"] += dvae._tpar["dvae_losses"]

        Path(MODEL_PATH / model_file_name).mkdir(exist_ok=True)
        with open(MODEL_PATH / model_file_name / "losses.json", "w") as file:
            json.dump(loss_data, file)

    mse_losses, dvae_losses = loss_data["mse_losses"], loss_data["dvae_losses"]

    fig_output = dvae.generate_output()
    if mse_losses and dvae_losses:
        fig_loss = dvae.generate_loss_plot(mse_losses, dvae_losses)

    fig_reconstructed = dvae.generate_reconstucted_samples()

    return fig_output, fig_loss, fig_reconstructed, "display-none"
