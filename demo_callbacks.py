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

import dash
from dash import MATCH
from dash.dependencies import Input, Output, State

from demo_interface import generate_problem_details_table_rows
from src.discrete_vae import run_generate, run_training


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
    Output("filename", "children"),
    Input("input-file", "filename"),
    prevent_initial_call=True,
)
def read_input_file(filename: str) -> str:
    """Display filename when file is selected.

    Args:
        filename: The name of the uploaded file.

    Returns:
        filename: The name of the file that was uploaded to display in the UI.
    """

    return filename


@dash.callback(
    Output("results", "children"),
    Output("problem-details", "children"),
    background=True,
    inputs=[
        Input("train-button", "n_clicks"),
        State("n-latents", "value"),
        State("file-name", "value"),
    ],
    running=[
        (Output("cancel-training-button", "className"), "", "display-none"),  # Show/hide cancel button.
        (Output("train-button", "className"), "display-none", ""),  # Hides run button while running.
        (Output("results-tab", "disabled"), True, False),  # Disables results tab while running.
        (Output("results-tab", "label"), "Loading...", "Results"),
        (Output("tabs", "value"), "input-tab", "input-tab"),  # Switch to input tab while running.
    ],
    cancel=[Input("cancel-training-button", "n_clicks")],
    prevent_initial_call=True,
)
def train(train_click: int, n_latents: int, file_name: str) -> tuple[str, list]:
    """Runs training and updates UI accordingly.

    This function is called when the ``Train`` button is clicked. It takes in all form values and
    runs the training, updates the run/cancel buttons, deactivates (and reactivates) the results
    tab, and updates all relevant HTML components.

    Args:
        train_click: The (total) number of times the train button has been clicked.
        n_latents: TODO
        file_name: TODO

    Returns:
        A tuple containing all outputs to be used when updating the HTML
        template (in ``demo_interface.py``). These are:

            results: The results to display in the results tab.
            problem-details: List of the table rows for the problem details table.
    """
    image = run_training(n_latents, file_name)

    # Generates a list of table rows for the problem details table.
    problem_details_table = generate_problem_details_table_rows(
        n_latents,
        file_name,
    )

    return (
        image,
        problem_details_table,
    )


@dash.callback(
    Output("results", "children", allow_duplicate=True),
    Output("problem-details", "children", allow_duplicate=True),
    background=True,
    inputs=[
        Input("generate-button", "n_clicks"),
        State("input-file", "filename"),
        State("tune-params", "value"),
        State("noise", "value"),
    ],
    running=[
        (Output("cancel-generation-button", "className"), "", "display-none"),  # Show/hide cancel button.
        (Output("generate-button", "className"), "display-none", ""),  # Hides run button while running.
        (Output("results-tab", "disabled"), True, False),  # Disables results tab while running.
        (Output("results-tab", "label"), "Loading...", "Results"),
        (Output("tabs", "value"), "input-tab", "input-tab"),  # Switch to input tab while running.
    ],
    cancel=[Input("cancel-generation-button", "n_clicks")],
    prevent_initial_call=True,
)
def generate(
    generate_click: int,
    training_file_name: str,
    tune_parameters: list,
    noise: float,
) -> tuple[str, list]:
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

            results: The results to display in the results tab.
            problem-details: List of the table rows for the problem details table.
    """
    image = run_generate(training_file_name, bool(len(tune_parameters)), noise)


    # Generates a list of table rows for the problem details table.
    problem_details_table = generate_problem_details_table_rows(training_file_name, tune_parameters)

    return (
        image,
        problem_details_table,
    )
