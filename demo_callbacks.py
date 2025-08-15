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
from pathlib import Path
from typing import NamedTuple

import dash
from dash import MATCH
from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate
from dwave.plugins.torch.models import DiscreteVariationalAutoencoder
from plotly import graph_objects as go
import plotly.io as pio

from demo_configs import SHARPEN_OUTPUT
from demo_interface import SOLVERS, generate_model_data, generate_options
from src.model_wrapper import ModelWrapper

MODEL_PATH = Path("models")
JSON_FILE_DIR = "generated_json"
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
    Output("model-details", "children"),
    Input("model-file-name", "value"),
)
def check_qpu_and_update_model(model_file_name: str) -> tuple[str, bool]:
    """Checks whether user has access to QPU associated with model and updates the model details
    when model changes.

    Args:
        model: The currently selected model

    Returns:
        popup-classname: The class name to hide the popup.
        generate-button-disabled: Whether to disable or enable the Generate button.
        model-details-children: The model details to display.
    """
    with open(MODEL_PATH / model_file_name / "parameters.json") as file:
        model_data = json.load(file)

    model_details = generate_model_data(model_data)

    if model_data["qpu"] and not (len(SOLVERS) and model_data["qpu"] in SOLVERS):
        return "", True, model_details

    return "display-none", False, model_details


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
    Input("last-trained-model", "data"),
)
def initialize_training_model(last_trained_model: str) -> tuple[list[str], str]:
    """Initializes the Trained Models dropdown options based on model files available.

    Args:
        last_trained_model: The most recently trained model directiory name.

    Returns:
        model-file-name-options: The options for the Trained Model dropdown selection.
        model-file-name-value: The value of the dropdown.
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

    return (
        models,
        last_trained_model if last_trained_model else models[0],
    )


@dash.callback(
    Output({"type": "progress-caption-epoch", "index": MATCH}, "children"),
    Output({"type": "progress-caption-batch", "index": MATCH}, "children"),
    Output({"type": "progress-wrapper", "index": MATCH}, "className"),
    inputs=[
        Input({"type": "progress", "index": MATCH}, "value"),
        State({"type": "progress", "index": MATCH}, "max"),
        State({"type": "n-epochs", "index": MATCH}, "value"),
    ],
    prevent_initial_call=True,
)
def update_progress(
    progress_value: str,
    progress_max: str,
    n_epochs: int,
) -> tuple[str, str, str]:
    """Updates progress bar with epochs and batches completed.

    Args:
        progress_value: The current value of the progress bar.
        progress_max: The maximum value of the progress bar ie progress_value/progress_max.
        n_epochs: The number of epochs to complete.

    Returns:
        progress-caption-epoch: The caption of the progress bar that tracks the completed epochs.
        progress-caption-batch: The caption of the progress bar that tracks the completed batches.
        progress-wrapper-className: The classname of the progress wrapper.
    """
    progress_value = int(progress_value) if progress_value else 0
    progress_max = int(progress_max) if progress_max else 0

    epoch_size = math.floor(progress_max / n_epochs)
    curr_epoch = math.floor(progress_value / epoch_size)

    return (
        f"Epochs Completed: {curr_epoch}/{n_epochs}",
        f"Batch: {progress_value%epoch_size}/{epoch_size}",
        "",
    )


@dash.callback(
    Output({"type": "progress-wrapper", "index": 0}, "className", allow_duplicate=True),
    Output({"type": "progress-wrapper", "index": 1}, "className", allow_duplicate=True),
    inputs=[
        Input("cancel-training-button", "n_clicks"),
        Input("cancel-generation-button", "n_clicks"),
    ],
    prevent_initial_call=True,
)
def cancel_progress(cancel_train: int, cancel_generate: int) -> tuple[str, str]:
    """Hides progress bar when cancel buttons are clicked.

    Args:
        cancel_train: The (total) number of times the train cancel button has been clicked.
        cancel_generate: The (total) number of times the generate cancel button has been clicked.

    Returns:
        progress-wrapper-className: The classname of the first progress wrapper.
        progress-wrapper-className: The classname of the second progress wrapper.
    """

    return "visibility-hidden", "visibility-hidden"


@dash.callback(
    Output("last-saved-id", "data", allow_duplicate=True),
    inputs=[
        Input("epoch-checker", "disabled"),
    ],
    prevent_initial_call=True,
)
def reset_last_saved_id(epoch_checker_disabled: bool) -> int:
    """Resets last-saved-id when epoch-checker interval is disabled.

    Args:
        epoch_checker_disabled: Whether the checker interval is disabled.

    Returns:
        last-saved-id: The id of the last saved file.
    """
    if epoch_checker_disabled:
        return None

    raise PreventUpdate


@dash.callback(
    Output("train-button", "disabled"),
    inputs=[
        Input("file-name", "value"),
    ],
)
def file_name_validation(file_name: str) -> bool:
    """Disables run button if no filename.

    Args:
        file_name: The value of the file name input.

    Returns:
        train-button-disabled: Whether the train button should be disabled.
    """
    return not file_name


class UpdateEachEpochReturn(NamedTuple):
    """Return type for the ``update_each_epoch`` callback function."""

    fig_generated: go.Figure = dash.no_update
    fig_reconstructed: go.Figure = dash.no_update
    fig_mse_loss: go.Figure = dash.no_update
    fig_total_loss: go.Figure = dash.no_update
    last_saved_id: int = dash.no_update
    results_tab_disabled: bool = dash.no_update
    loss_tab_disabled: bool = dash.no_update
    tabs_value: str = dash.no_update

@dash.callback(
    Output("fig-output", "figure", allow_duplicate=True),
    Output("fig-reconstructed", "figure", allow_duplicate=True),
    Output("fig-mse-loss", "figure", allow_duplicate=True),
    Output("fig-total-loss", "figure", allow_duplicate=True),
    Output("last-saved-id", "data"),
    Output("results-tab", "disabled"),
    Output("loss-tab", "disabled"),
    Output("tabs", "value"),
    inputs=[
        Input("epoch-checker", "n_intervals"),
        State("last-saved-id", "data"),
    ],
    prevent_initial_call=True,
)
def update_each_epoch(
    epoch_checker: int, last_saved_id: int
) -> UpdateEachEpochReturn:
    """Updates visuals after each epoch.

    Args:
        epoch_checker: An interval that fires to check whether new files have been generated.
        last_saved_id: The ID of the file that was last saved.
        
    Returns:
        UpdateEachEpochReturn named tuple:
            fig_generated: The generated image output.
            fig_reconstructed: The image comparing the reconstructed image to the original.
            fig_mse_loss: The graph showing the MSE Loss.
            fig_total_loss: The graph showing the total Loss (MMD + MSE).
            last_saved_id: The ID of the file that was last saved.
            results_tab_disabled: Whether the results tab should be disabled.
            loss_tab_disabled: Whether the loss tab should be disabled.
            tabs_value: The tab that should be active.
    """
    
    if last_saved_id is None:
        json_path = Path(JSON_FILE_DIR)
        json_path.mkdir(exist_ok=True)
        for file in json_path.iterdir():
            file.unlink()  # Delete all files on first iteration.

        return UpdateEachEpochReturn(
            last_saved_id=0,
            results_tab_disabled=True,
            loss_tab_disabled=True,
            tabs_value="input-tab",
        )

    new_file_id = last_saved_id+1
    image_gen_file_path = f"{JSON_FILE_DIR}/{IMAGE_GEN_FILE_PREFIX}{new_file_id}.json"
    image_recon_file_path = f"{JSON_FILE_DIR}/{IMAGE_RECON_FILE_PREFIX}{new_file_id}.json"
    loss_mse_file_path = f"{JSON_FILE_DIR}/{LOSS_PREFIX}mse_{new_file_id}.json"
    loss_total_file_path = f"{JSON_FILE_DIR}/{LOSS_PREFIX}total_{new_file_id}.json"

    try:
        with open(image_gen_file_path, "r") as f:
            fig_gen_json = json.load(f)
            fig_gen = pio.from_json(json.dumps(fig_gen_json))
        with open(image_recon_file_path, "r") as f:
            fig_recon_json = json.load(f)
            fig_recon = pio.from_json(json.dumps(fig_recon_json))
        with open(loss_mse_file_path, "r") as f:
            fig_mse_json = json.load(f)
            fig_mse = pio.from_json(json.dumps(fig_mse_json))
        with open(loss_total_file_path, "r") as f:
            fig_total_json = json.load(f)
            fig_total = pio.from_json(json.dumps(fig_total_json))

        return UpdateEachEpochReturn(
            fig_generated=fig_gen,
            fig_reconstructed=fig_recon,
            fig_mse_loss=fig_mse,
            fig_total_loss=fig_total,
            last_saved_id=new_file_id,
            results_tab_disabled=False,
            loss_tab_disabled=False,
        )

    except:
        # No file found, this is expected behavior before the epoch has finished.
        raise PreventUpdate
    


@dash.callback(
    Output("fig-output", "figure", allow_duplicate=True),
    Output("fig-mse-loss", "figure", allow_duplicate=True),
    Output("fig-total-loss", "figure", allow_duplicate=True),
    Output("fig-reconstructed", "figure", allow_duplicate=True),
    Output("last-trained-model", "data"),
    Output({"type": "progress-wrapper", "index": 0}, "className", allow_duplicate=True),
    background=True,
    inputs=[
        Input("train-button", "n_clicks"),
        State("qpu-setting", "value"),
        State("n-latents", "value"),
        State({"type": "n-epochs", "index": 0}, "value"),
        State("file-name", "value"),
    ],
    running=[
        (Output("cancel-training-button", "className"), "", "display-none"),
        (Output("train-button", "className"), "display-none", ""),
        (Output("generate-tab", "disabled"), True, False),  # Disables generate tab while running.
        (Output("results-tab", "label"), "Loading...", "Generated Images"),
        (Output("loss-tab", "label"), "Loading...", "Loss Graphs"),
        (Output("epoch-checker", "disabled"), False, True),
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
            fig-mse-loss: The graph showing the MSE Loss.
            fig-total-loss: The graph showing the total Loss (MMD + MSE).
            fig-reconstructed: The image comparing the reconstructed image to the original.
            last-trained-model: The directory name of the model trained by this run.
            progress-wrapper-className: The classname of the progress wrapper.
    """
    model = ModelWrapper(qpu=qpu, n_latents=n_latents)

    model.train_init(n_epochs)

    for epoch in range(n_epochs):
        start_time = time.perf_counter()
        print(f"Starting epoch {epoch + 1}/{n_epochs}")

        total = len(model._dataloader)
        for i, batch in enumerate(model._dataloader):
            set_progress((str(total * epoch + i), str(total * n_epochs)))
            mse_loss = model.step(batch, epoch)

        print(
            f"Epoch {epoch + 1}/{n_epochs} - MSE Loss: {mse_loss.item():.4f} - "
            f'Learning rate DVAE: {model._tpar["dvae_lr_schedule"][model._tpar["opt_step"]]:.3E} '
            f'Learning rate GRBM: {model._tpar["grbm_lr_schedule"][model._tpar["opt_step"]]:.3E} '
            f"Time: {(time.perf_counter() - start_time)/60:.2f} mins. "
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
        )

    create_model_files(
        model,
        file_name,
        qpu,
        n_latents,
        n_epochs,
        {
            "mse_losses": model.losses["mse_losses"],
            "dvae_losses": model.losses["dvae_losses"],
        },
    )

    return (
        fig_output,
        fig_mse_loss,
        fig_dvae_loss,
        fig_reconstructed,
        file_name,
        "visibility-hidden",
    )


@dash.callback(
    Output("fig-output", "figure"),
    Output("fig-mse-loss", "figure"),
    Output("fig-total-loss", "figure"),
    Output("fig-reconstructed", "figure"),
    Output("popup", "className", allow_duplicate=True),
    Output({"type": "progress-wrapper", "index": 1}, "className", allow_duplicate=True),
    background=True,
    inputs=[
        Input("generate-button", "n_clicks"),
        State("model-file-name", "value"),
        State("tune-params", "value"),
        State({"type": "n-epochs", "index": 1}, "value"),
    ],
    running=[
        (Output("cancel-generation-button", "className"), "", "display-none"),
        (Output("generate-button", "className"), "display-none", ""),
        (Output("results-tab", "disabled"), True, False),  # Disables results tab while running.
        (Output("loss-tab", "disabled"), True, False),  # Disables loss tab while running.
        (Output("train-tab", "disabled"), True, False),  # Disables train tab while running.
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
            fig-mse-loss: The graph showing the MSE Loss.
            fig-total-loss: The graph showing the total Loss (MMD + MSE).
            fig-reconstructed: The image comparing the reconstructed image to the original.
            progress-wrapper-className: The classname of the progress wrapper.
    """
    # load autoencoder model and config
    with open(MODEL_PATH / model_file_name / "parameters.json") as file:
        model_data = json.load(file)
    with open(MODEL_PATH / model_file_name / "losses.json") as file:
        loss_data = json.load(file)

    if model_data["qpu"] and not (len(SOLVERS) and model_data["qpu"] in SOLVERS):
        return dash.no_update, dash.no_update, dash.no_update, "", "visibility-hidden"

    model = ModelWrapper(qpu=model_data["qpu"], n_latents=model_data["n_latents"])
    model.load(file_path=MODEL_PATH / model_file_name)

    if tune_parameters:
        model.train_init(n_epochs)

        for epoch in range(n_epochs):
            start_time = time.perf_counter()
            print(f"Starting epoch {epoch + 1}/{n_epochs}")

            total = len(model._dataloader)
            for i, batch in enumerate(model._dataloader):
                print(f"{i}/{total}")
                set_progress((str(total * epoch + i), str(total * n_epochs)))
                mse_loss = model.step(batch, epoch)

            print(
                f"Epoch {epoch + 1}/{n_epochs} - MSE Loss: {mse_loss.item():.4f} - "
                f'Learning rate DVAE: {model._tpar["dvae_lr_schedule"][model._tpar["opt_step"]]:.3E} '
                f'Learning rate GRBM: {model._tpar["grbm_lr_schedule"][model._tpar["opt_step"]]:.3E} '
                f"Time: {(time.perf_counter() - start_time)/60:.2f} mins. "
            )

        model_file_name += f"_tuned_{n_epochs}_epochs"

        loss_data["mse_losses"] += model.losses["mse_losses"]
        loss_data["dvae_losses"] += model.losses["dvae_losses"]

        Path(MODEL_PATH / model_file_name).mkdir(exist_ok=True)

        create_model_files(
            model, model_file_name, model_data["qpu"], model_data["n_latents"], n_epochs, loss_data
        )

    fig_output = model.generate_output(sharpen=SHARPEN_OUTPUT)

    model.losses = loss_data
    fig_mse_loss, fig_dvae_loss = model.generate_loss_plot()

    fig_reconstructed = model.generate_reconstucted_samples(sharpen=SHARPEN_OUTPUT)

    return (
        fig_output,
        fig_mse_loss,
        fig_dvae_loss,
        fig_reconstructed,
        "display-none",
        "visibility-hidden",
    )
