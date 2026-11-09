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
import logging
import math
import random
import re
from pathlib import Path
from typing import NamedTuple
import torch

import dash
import plotly.io as pio
from dash import ALL, ctx, html, MATCH
from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate
from plotly import graph_objects as go
from torchvision.utils import make_grid

from demo_configs import FFHQ_MODEL_NAME, GENERATE_NEW_MODEL_DIAGRAM, GRAPH_COLORS
from demo_interface import (
    SOLVERS,
    generate_latent_vector,
    generate_model_data,
    generate_problem_details_table
)
from src.model_wrapper import ModelWrapper
from src.generation import get_engine, image_data_uri
from src.utils.callback_helpers import (
    IMAGE_GEN_FILE_PREFIX,
    IMAGE_RECON_FILE_PREFIX,
    JSON_FILE_DIR,
    LATENT_ENCODED_FILE,
    LATENT_QPU_FILE,
    LOSS_PREFIX,
    MODEL_PATH,
    PROBLEM_DETAILS_PATH,
    STEP_2_FILE,
    STEP_4_FILE,
    STEP_5_FILE,
    create_model_files,
    execute_training,
    generate_model_diagram,
    generate_model_fig,
)


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
    Output("step-2-encode-img", "src", allow_duplicate=True),
    Output("step-4-decode-img", "src", allow_duplicate=True),
    Output("step-5-output-img", "src", allow_duplicate=True),
    Output("fig-qpu-graph", "figure", allow_duplicate=True),
    Output("fig-encoded-graph", "figure", allow_duplicate=True),
    Output("latent-space-vector", "children"),
    inputs=[
        Input({"type": "progress", "index": ALL}, "value"),
        State("fig-qpu-graph", "figure"),
        State("fig-encoded-graph", "figure"),
        State("latent-mapping", "data"),
    ],
    prevent_initial_call=True,
)
def update_model_diagram_imgs(
    progress: int,
    fig_qpu: go.Figure,
    fig_encoded: go.Figure,
    latent_mapping: list,
) -> tuple[str, str, str, go.Figure, go.Figure, list]:
    """Force refresh images to get around Dash caching. Updates image src with a incrementing
    query string.

    Args:
        progress: The train or tune progress status.
        fig_qpu: The QPU graph figure.
        fig_encoded: The not QPU graph figure.
        latent_mapping: The mapping of the nodes to latent space indices.

    Returns:
        step-2-encode-img: The file path for the encode image.
        step-4-decode-img: The file path for the decode image.
        step-5-output-img: The file path for the output image.
        fig-qpu-graph: The QPU graph figure.
        fig-encoded-graph: The not QPU graph figure.
        latent-space-vector: The Dash HTML for the plus and minus ones visual vector.
    """
    if not GENERATE_NEW_MODEL_DIAGRAM:
        raise PreventUpdate

    fig_qpu = go.Figure(fig_qpu)
    fig_encoded = go.Figure(fig_encoded)

    with open(LATENT_QPU_FILE, "r") as f:
        latent_qpu = json.load(f)

    with open(LATENT_ENCODED_FILE, "r") as f:
        latent_encoded = json.load(f)

    color_mapping_qpu = [GRAPH_COLORS[int(latent_qpu[i] > 0)] for i in latent_mapping]
    color_mapping_encoded = [GRAPH_COLORS[int(latent_encoded[i] > 0)] for i in latent_mapping]

    fig_qpu.update_traces(marker=dict(color=color_mapping_qpu))
    fig_encoded.update_traces(marker=dict(color=color_mapping_encoded))

    return (
        f"{STEP_2_FILE}?interval={progress}",
        f"{STEP_4_FILE}?interval={progress}",
        f"{STEP_5_FILE}?interval={progress}",
        fig_qpu,
        fig_encoded,
        generate_latent_vector(latent_encoded[:5], latent_encoded[-1]),
    )


class CheckQpuAndUpdateModelReturn(NamedTuple):
    """Return type for the ``check_qpu_and_update_model`` callback function."""

    popup_classname: str = "display-none"
    generate_button_disabled: bool = False
    model_details: dict = dash.no_update
    fig_qpu_graph: go.Figure = dash.no_update
    fig_encoded_graph: go.Figure = dash.no_update
    latent_diagram_size: int = dash.no_update
    latent_mapping: list[int] = dash.no_update
    step_2_img: str = dash.no_update
    step_4_img: str = dash.no_update
    step_5_img: str = dash.no_update
    has_loaded_diagram: bool = True

@dash.callback(
    Output("popup", "className"),
    Output("generate-button", "disabled"),
    Output("model-details", "children"),
    Output("fig-qpu-graph", "figure"),
    Output("fig-encoded-graph", "figure"),
    Output("latent-diagram-size", "children"),
    Output("latent-mapping", "data"),
    Output("step-2-encode-img", "src"),
    Output("step-4-decode-img", "src"),
    Output("step-5-output-img", "src"),
    Output("has-loaded-diagram", "data"),
    inputs=[
        Input("model-file-name", "value"),
        Input("inference-qpu", "value"),
        Input("n-latents", "value"),
        Input("setting-tabs", "value"),
        State("example-image", "data"),
        State("has-loaded-diagram", "data"),
    ]
)
def check_qpu_and_update_model(
    model_file_name: str,
    qpu: str,
    n_latents: int,
    setting_tabs_value: str,
    example_image: list,
    has_loaded_diagram: bool,
) -> CheckQpuAndUpdateModelReturn:
    """Checks whether user has access to QPU associated with model and updates the model details
    when model changes.

    Args:
        model: The currently selected model.
        qpu: The selected QPU.
        n_latents: The dimension of the latent space.
        setting_tabs_value: The currently selected settings tab.
        example_image: The example image to show all the steps for in the UI.
        has_loaded_diagram: Keeps track of whether this is a page load.

    Returns:
        CheckQpuAndUpdateModelReturn named tuple:
            popup_classname: The class name to hide the popup.
            generate_button_disabled: Whether to disable or enable the Generate button.
            model_details: The model details to display.
            fig_qpu_graph: The QPU graph figure.
            fig_encoded_graph: The not QPU graph figure.
            latent_diagram_size: The dimension of the latent space.
            latent_mapping: The mapping of the nodes to latent space indices.
            step_2_img: The file path for the encode image.
            step_4_img: The file path for the decode image.
            step_5_img: The file path for the output image.
            has_loaded_diagram: Keeps track of whether this is a page load.
    """
    if model_file_name == FFHQ_MODEL_NAME:
        return CheckQpuAndUpdateModelReturn(
            generate_button_disabled=qpu not in SOLVERS,
            model_details=html.P("Pretrained FFHQ decoder; inference only."),
        )

    switched_to_generate_tab = ctx.triggered_id == "setting-tabs" and setting_tabs_value == "generate-tab"

    # If first load, or a new model is chosen, or the settings tab is changed to "generate"
    if not ctx.triggered_id or ctx.triggered_id == "model-file-name" or switched_to_generate_tab:
        with open(MODEL_PATH / model_file_name / "parameters.json") as file:
            model_data = json.load(file)

        model_details = generate_model_data(model_data)

        # If model_data has a QPU that is no longer available, show warning popup
        if model_data["qpu"] and not (len(SOLVERS) and model_data["qpu"] in SOLVERS):
            return CheckQpuAndUpdateModelReturn(
                popup_classname="",
                generate_button_disabled=True,
                model_details=model_details,
            )

        # Create model instance to generate model diagram images
        model = ModelWrapper(qpu=model_data["qpu"], n_latents=model_data["n_latents"])
        model.load(file_path=MODEL_PATH / model_file_name)

        # Dash converts the tensor to a list of floats, convert back to tensor
        example_image = torch.tensor(example_image, dtype=torch.float32)
        example_image = example_image.unsqueeze(0)
        generate_model_diagram(model, example_image)

        fig_qpu, fig_encoded, latent_mapping = generate_model_fig(
            model_data["qpu"],
            model_data["n_latents"],
            model_data["random_seed"],
        )

        force_refresh = random.randint(1, 9999999)

        return CheckQpuAndUpdateModelReturn(
            model_details=model_details,
            fig_qpu_graph=fig_qpu,
            fig_encoded_graph=fig_encoded,
            latent_diagram_size=model_data["n_latents"],
            latent_mapping=latent_mapping,
            step_2_img=f"{STEP_2_FILE}?force_refresh={force_refresh}",
            step_4_img=f"{STEP_4_FILE}?force_refresh={force_refresh}",
            step_5_img=f"{STEP_5_FILE}?force_refresh={force_refresh}" if has_loaded_diagram else dash.no_update,
            has_loaded_diagram=False if not ctx.triggered_id else True,
        )

    # No model data, proceed with defaults
    fig_qpu, fig_encoded, latent_mapping = generate_model_fig(qpu, n_latents, 4)

    return CheckQpuAndUpdateModelReturn(
        fig_qpu_graph=fig_qpu,
        fig_encoded_graph=fig_encoded,
        latent_diagram_size=n_latents,
        latent_mapping=latent_mapping,
    )


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
    """Selects the bundled FFHQ checkpoint without loading legacy training models.

    Args:
        last_trained_model: The most recently trained model directiory name.

    Returns:
        model-file-name-options: The options for the Trained Model dropdown selection.
        model-file-name-value: The value of the dropdown.
    """
    return (
        [{"label": "FFHQ EMA (step 540,000)", "value": FFHQ_MODEL_NAME}],
        FFHQ_MODEL_NAME,
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
    Output("file-name-help-text", "className"),
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
        file-name-help-text-classname: Whether to hide or show the file name help text.
    """
    if not file_name:
        return True, "display-none"

    pattern = re.compile("^[\\w\\-]+$")  # allows for a-z A-Z 0-9 _ -
    is_valid = pattern.match(file_name)

    return not is_valid, "display-none" if is_valid else ""


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
    results_tab_label: str = dash.no_update
    loss_tab_label: str = dash.no_update
    problem_details_table: list = dash.no_update


@dash.callback(
    Output("fig-output", "figure", allow_duplicate=True),
    Output("fig-reconstructed", "figure", allow_duplicate=True),
    Output("fig-mse-loss", "figure", allow_duplicate=True),
    Output("fig-total-loss", "figure", allow_duplicate=True),
    Output("last-saved-id", "data"),
    Output("results-tab", "disabled"),
    Output("loss-tab", "disabled"),
    Output("tabs", "value"),
    Output("results-tab", "label"),
    Output("loss-tab", "label"),
    Output("problem-details", "children"),
    inputs=[
        Input("epoch-checker", "n_intervals"),
        State("last-saved-id", "data"),
    ],
    prevent_initial_call=True,
)
def update_each_epoch(epoch_checker: int, last_saved_id: int) -> UpdateEachEpochReturn:
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
            results_tab_label: The label for the results tab.
            loss_tab_label: The label for the loss tab.
            problem_details_table: The html for a table outlining the details each epoch.
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

    new_file_id = last_saved_id + 1
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
        with open(PROBLEM_DETAILS_PATH, "r") as f:
            problem_details = json.load(f)

        return UpdateEachEpochReturn(
            fig_generated=fig_gen,
            fig_reconstructed=fig_recon,
            fig_mse_loss=fig_mse,
            fig_total_loss=fig_total,
            last_saved_id=new_file_id,
            results_tab_disabled=False,
            loss_tab_disabled=False,
            results_tab_label=f"Generated Images (after {new_file_id} epoch{'s'[:new_file_id^1]})",
            loss_tab_label=f"Loss Graphs (after {new_file_id} epoch{'s'[:new_file_id^1]})",
            problem_details_table=generate_problem_details_table(problem_details),
        )

    except:
        # No file found, this is expected behavior before the epoch has finished.
        raise PreventUpdate


@dash.callback(
    Output("fig-output", "figure", allow_duplicate=True),
    Output("fig-reconstructed", "figure", allow_duplicate=True),
    Output("fig-mse-loss", "figure", allow_duplicate=True),
    Output("fig-total-loss", "figure", allow_duplicate=True),
    Output("last-trained-model", "data"),
    Output({"type": "progress-wrapper", "index": 0}, "className", allow_duplicate=True),
    background=True,
    inputs=[
        Input("train-button", "n_clicks"),
        State("qpu-setting", "value"),
        State("n-latents", "value"),
        State({"type": "n-epochs", "index": 0}, "value"),
        State("file-name", "value"),
        State("example-image", "data"),
    ],
    running=[
        (Output("cancel-training-button", "className"), "", "display-none"),
        (Output("train-button", "className"), "display-none", ""),
        (Output("generate-tab", "disabled"), True, False),  # Disables generate tab while running.
        (Output("results-tab", "label"), "Training...", "Generated Images"),
        (Output("loss-tab", "label"), "Training...", "Loss Graphs"),
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
    set_progress,
    train_click: int,
    qpu: str,
    n_latents: int,
    n_epochs: int,
    file_name: str,
    example_image: list,
) -> tuple[go.Figure, go.Figure, go.Figure, go.Figure, str, str]:
    """Runs training and updates UI accordingly.

    This function is called when the ``Train`` button is clicked. It takes in all form values and
    runs the training, updates the run/cancel buttons, deactivates (and reactivates) the results
    tab, and updates all relevant HTML components.

    Args:
        train_click: The (total) number of times the train button has been clicked.
        qpu: The selected QPU.
        n_latents: The value of the latents setting.
        n_epochs: The value of the epochs setting.
        file_name: The file name to save to.
        example_image: The example image to show all the steps for in the UI.

    Returns:
        A tuple containing all outputs to be used when updating the HTML
        template (in ``demo_interface.py``). These are:

            fig-output: The generated image output.
            fig-reconstructed: The image comparing the reconstructed image to the original.
            fig-mse-loss: The graph showing the MSE Loss.
            fig-total-loss: The graph showing the total Loss (MMD + MSE).
            last-trained-model: The directory name of the model trained by this run.
            progress-wrapper-className: The classname of the progress wrapper.
    """
    raise PreventUpdate  # Training stays disabled until the legacy path is removed.

    model = ModelWrapper(qpu=qpu, n_latents=n_latents)

    # Dash converts the tensor to a list of floats, convert back to tensor
    example_image = torch.tensor(example_image, dtype=torch.float32)

    model.train_init(n_epochs)
    fig_output, fig_reconstructed, fig_mse_loss, fig_dvae_loss = execute_training(
        set_progress, model, n_epochs, qpu, n_latents, example_image=example_image
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
        fig_reconstructed,
        fig_mse_loss,
        fig_dvae_loss,
        file_name,
        "visibility-hidden",
    )


class GenerateReturn(NamedTuple):
    """Return type for the ``generate`` callback function."""

    fig_generated: go.Figure = dash.no_update
    fig_reconstructed: go.Figure = dash.no_update
    fig_mse_loss: go.Figure = dash.no_update
    fig_total_loss: go.Figure = dash.no_update
    popup_classname: str = "display-none"
    progress_wrapper_classname: str = "visibility-hidden"
    results_tab_disabled: bool = dash.no_update
    loss_tab_disabled: bool = dash.no_update
    problem_details_table: list = dash.no_update
    tabs_value: str = "results-tab"


@dash.callback(
    Output("fig-output", "figure"),
    Output("fig-reconstructed", "figure"),
    Output("fig-mse-loss", "figure"),
    Output("fig-total-loss", "figure"),
    Output("popup", "className", allow_duplicate=True),
    Output({"type": "progress-wrapper", "index": 1}, "className", allow_duplicate=True),
    Output("results-tab", "disabled", allow_duplicate=True),
    Output("loss-tab", "disabled", allow_duplicate=True),
    Output("problem-details", "children", allow_duplicate=True),
    Output("tabs", "value", allow_duplicate=True),
    inputs=[
        Input("generate-button", "n_clicks"),
        State("model-file-name", "value"),
        State("inference-qpu", "value"),
    ],
    running=[
        (Output("generate-button", "disabled", allow_duplicate=True), True, False),
        (Output("inference-qpu", "disabled"), True, False),
        (Output("model-file-name", "disabled"), True, False),
        (Output("results-tab", "label"), "Generating...", "Generated Images"),
    ],
    prevent_initial_call=True,
)
def generate(
    generate_click: int,
    model_file_name: str,
    qpu: str,
) -> GenerateReturn:
    """Generate four selected FFHQ faces using cached, read-only inference models."""
    try:
        if model_file_name != FFHQ_MODEL_NAME:
            raise ValueError("Select the bundled FFHQ inference model.")
        result = get_engine().generate(qpu)
        grid = make_grid(result.images, nrow=2, padding=0)
        fig_output = go.Figure(go.Image(source=image_data_uri(grid)))
        fig_output.update_xaxes(showticklabels=False)
        fig_output.update_yaxes(showticklabels=False)
        fig_output.update_layout(margin={"t": 0, "l": 0, "b": 0, "r": 0})
    except Exception:
        logging.getLogger(__name__).exception("FFHQ generation failed on %s.", qpu)
        return GenerateReturn(
            fig_generated=go.Figure(),
            results_tab_disabled=False,
            loss_tab_disabled=True,
            problem_details_table=[
                html.P("Generation failed. See the server log; no substitute samples were used.")
            ],
        )

    info = result.metadata
    return GenerateReturn(
        fig_generated=fig_output,
        results_tab_disabled=False,
        loss_tab_disabled=True,
        problem_details_table=generate_problem_details_table({
            "QPU": info["solver"],
            "Embedding": info["embedding_id"],
            "Candidates": info["candidates"],
            "EMA step": info["checkpoint_step"],
            "Seconds": f"{info['seconds']:.1f}",
            "Quality ranks": ", ".join(map(str, result.quality_ranks)),
        }),
    )
