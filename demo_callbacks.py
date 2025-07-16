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
import os
import time
from pathlib import Path

import dash
from dash import MATCH
from dash.dependencies import Input, Output, State
from plotly import graph_objects as go

from demo_interface import generate_options
from src.model_wrapper import ModelWrapper


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
    Output("tune-parameter-settings", "className"),
    Input("tune-params", "value"),
)
def toggle_tuning_params(tune_params: list[int]) -> str:
    """Show/hide tune parameter settings when Tune Parameters box is toggled.

    Args:
        tune_params: The value of the Tune Paraters checkbox as a list.

    Returns:
        tune-parameter-settings-classname: The class name to show/hide the tune parameter settings.
    """
    if len(tune_params):
        return ""

    return "display-none"


@dash.callback(
    Output("model-file-name", "options"),
    Output("model-file-name", "value"),
    Input("fig-output", "figure"),
)
###TODO make trigger when training finishes
def render_initial_state(fig: go.Figure) -> str:
    """TODO

    Args:
        TODO

    Returns:
        TODO
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

    return models, models[0]


@dash.callback(
    Output("problem-details", "children"),
    background=True,
    inputs=[
        Input("train-button", "n_clicks"),
        State("n-latents", "value"),
        State("n-epochs", "value"),
        State("file-name", "value"),
    ],
    running=[
        (
            Output("cancel-training-button", "className"),
            "",
            "display-none",
        ),  # Show/hide cancel button.
        (
            Output("train-button", "className"),
            "display-none",
            "",
        ),  # Hides run button while running.
        (Output("results-tab", "disabled"), True, False),  # Disables results tab while running.
        (Output("loss-tab", "disabled"), True, False),  # Disables loss tab while running.
        (Output("results-tab", "label"), "Loading...", "Generated Images"),
        (Output("loss-tab", "label"), "Loading...", "Loss"),
        (Output("tabs", "value"), "input-tab", "input-tab"),  # Switch to input tab while running.
        (Output("batch-progress", "style"), {"visibility": "visible"}, {"visibility": "hidden"}),
        (Output("epoch-progress", "style"), {"visibility": "visible"}, {"visibility": "hidden"}),
    ],
    cancel=[Input("cancel-training-button", "n_clicks")],
    progress=[
        Output("batch-progress", "value"),
        Output("batch-progress", "max"),
        Output("epoch-progress", "value"),
        Output("epoch-progress", "max"),
    ],
    prevent_initial_call=True,
)
def train(
    set_progress, train_click: int, n_latents: int, n_epochs: int, file_name: str
) -> tuple[go.Figure, go.Figure, go.Figure]:
    """Runs training and updates UI accordingly.

    This function is called when the ``Train`` button is clicked. It takes in all form values and
    runs the training, updates the run/cancel buttons, deactivates (and reactivates) the results
    tab, and updates all relevant HTML components.

    Args:
        train_click: The (total) number of times the train button has been clicked.
        n_latents: TODO
        n_epochs: TODO
        file_name: TODO

    Returns:
        A tuple containing all outputs to be used when updating the HTML
        template (in ``demo_interface.py``). These are:

            fig_output: TODO
            fig_loss: TODO
            fig_reconstructed: TODO
    """
    model_path = Path("models")

    dvae = ModelWrapper(n_latents=n_latents)

    dvae.train_init(n_epochs, perturb_grbm=False)

    for epoch in range(n_epochs):
        start_time = time.perf_counter()
        print(f"Starting epoch {epoch + 1}/{n_epochs}")

        total = len(dvae._dataloader)
        for i, batch in enumerate(dvae._dataloader):
            set_progress((str(i), str(total), str(epoch), str(n_epochs)))
            mse_loss = dvae.step(batch, epoch)

        print(
            f"Epoch {epoch + 1}/{n_epochs} - MSE Loss: {mse_loss.item():.4f} - "
            f'Learning rate DVAE: {dvae._tpar["dvae_lr_schedule"][dvae._tpar["opt_step"]]:.3E} '
            f'Learning rate GRBM: {dvae._tpar["grbm_lr_schedule"][dvae._tpar["opt_step"]]:.3E} '
            f"Time: {(time.perf_counter() - start_time)/60:.2f} mins. "
        )

    dvae.save(file_path=model_path / file_name)

    with open(model_path / file_name / "parameters.json", "w") as f:
        json.dump(
            {
                "n_latents": n_latents,
                "use_qpu": dvae.USE_QPU,
                "num_read": dvae.NUM_READS,
                "loss_function": dvae.LOSS_FUNCTION,
                "image_size": dvae.IMAGE_SIZE,
                "batch_size": dvae.BATCH_SIZE,
                "dateset_size": dvae.DATASET_SIZE,
            },
            f,
        )

    with open(model_path / file_name / "losses.json", "w") as f:
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

    return fig_output, fig_loss, fig_reconstructed


@dash.callback(
    Output("fig-output", "figure"),
    Output("fig-loss", "figure"),
    Output("fig-reconstructed", "figure"),
    background=True,
    inputs=[
        Input("generate-button", "n_clicks"),
        State("model-file-name", "value"),
        State("tune-params", "value"),
        State("noise", "value"),
        State("n-epochs-tune", "value"),
    ],
    running=[
        (
            Output("cancel-generation-button", "className"),
            "",
            "display-none",
        ),  # Show/hide cancel button.
        (
            Output("generate-button", "className"),
            "display-none",
            "",
        ),  # Hides run button while running.
        (Output("results-tab", "disabled"), True, False),  # Disables results tab while running.
        (Output("loss-tab", "disabled"), True, False),  # Disables loss tab while running.
        (Output("results-tab", "label"), "Loading...", "Generated Images"),
        (Output("loss-tab", "label"), "Loading...", "Loss"),
        (Output("tabs", "value"), "input-tab", "input-tab"),  # Switch to input tab while running.
        (Output("tune-progress", "style"), {"visibility": "visible"}, {"visibility": "hidden"}),
    ],
    progress=[
        Output("tune-progress", "value"),
        Output("tune-progress", "max"),
    ],
    cancel=[Input("cancel-generation-button", "n_clicks")],
    prevent_initial_call=True,
)
def generate(
    set_progress,
    generate_click: int,
    training_file_name: str,
    tune_parameters: list,
    noise: float,
    n_epochs: int,
) -> tuple[go.Figure, go.Figure, go.Figure]:
    """Runs generation and updates UI accordingly.

    This function is called when the ``Generate`` button is clicked. It takes in all form values and
    runs the generation, updates the run/cancel buttons, deactivates (and reactivates) the results
    tab, and updates all relevant HTML components.

    Args:
        generate_click: The (total) number of times the generate button has been clicked.
        training_file_name: TODO
        tune_parameters: TODO
        noise: TODO

    Returns:
        A tuple containing all outputs to be used when updating the HTML
        template (in ``demo_interface.py``). These are:

            fig_output: TODO
            fig_loss: TODO
            fig_reconstructed: TODO
    """
    model_path = Path("models")

    # load autoencoder model and config
    with open(model_path / training_file_name / "parameters.json") as file:
        model_data = json.load(file)
    with open(model_path / training_file_name / "losses.json") as file:
        loss_data = json.load(file)

    dvae = ModelWrapper(n_latents=model_data["n_latents"])
    dvae.load(file_path=model_path / training_file_name)

    if tune_parameters:
        dvae.train_init(n_epochs, perturb_grbm=False, noise=noise)

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

        training_file_name += f"_tuned_{n_epochs}"

        loss_data["mse_losses"] += dvae._tpar["mse_losses"]
        loss_data["dvae_losses"] += dvae._tpar["dvae_losses"]

        Path(model_path / training_file_name).mkdir(exist_ok=True)
        with open(model_path / training_file_name / "losses.json", "w") as file:
            json.dump(loss_data, file)

    mse_losses, dvae_losses = loss_data["mse_losses"], loss_data["dvae_losses"]

    fig_output = dvae.generate_output()
    if mse_losses and dvae_losses:
        fig_loss = dvae.generate_loss_plot(mse_losses, dvae_losses)

    fig_reconstructed = dvae.generate_reconstucted_samples()

    return fig_output, fig_loss, fig_reconstructed
