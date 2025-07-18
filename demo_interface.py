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

"""This file stores the Dash HTML layout for the app."""
from __future__ import annotations
from typing import Any, Optional

from dash import dcc, html
from plotly import graph_objects as go

from demo_configs import (
    DEFAULT_QPU,
    DESCRIPTION,
    MAIN_HEADER,
    NOISE,
    SLIDER_EPOCHS,
    SLIDER_LATENTS,
    THUMBNAIL,
)
from src.model_wrapper import display_dataset, get_dataset
from dwave.cloud import Client

ANNEAL_TIME_RANGES = {}

# Initialize: available QPUs
try:
    client = Client.from_config(client="qpu")
    SOLVERS = [qpu.name for qpu in client.get_solvers()]

    if not len(SOLVERS):
        raise Exception

except Exception:
    SOLVERS = ["No Leap Access"]


def display_input_data() -> go.Figure:
    """Load data from MNIST and display in input tab.

    Returns:
        fig: a figure of MNIST data.
    """
    dataset = get_dataset(32, 32*22)
    fig = display_dataset(dataset, 32)

    return fig


def slider(label: str, id: str, config: dict) -> html.Div:
    """Slider element for value selection.

    Args:
        label: The title that goes above the slider.
        id: A unique selector for this element.
        config: A dictionary of slider configerations, see dcc.Slider Dash docs.
    """
    return html.Div(
        className="slider-wrapper",
        children=[
            html.Label(label),
            dcc.Slider(
                id=id,
                className="slider",
                **config,
                marks={
                    config["min"]: str(config["min"]),
                    config["max"]: str(config["max"]),
                },
                tooltip={
                    "placement": "bottom",
                    "always_visible": True,
                },
            ),
        ],
    )


def dropdown(label: str, id: str, options: list, value: Optional[Any] = None) -> html.Div:
    """Dropdown element for option selection.

    Args:
        label: The title that goes above the dropdown.
        id: A unique selector for this element.
        options: A list of dictionaries of labels and values.
        value: Optional default value.
    """
    return html.Div(
        className="dropdown-wrapper",
        children=[
            html.Label(label),
            dcc.Dropdown(
                id=id,
                options=options,
                value=value if value else options[0]["value"],
                clearable=False,
                searchable=False,
            ),
        ],
    )


def checklist(label: str, id: str, options: list, values: list, inline: bool = True) -> html.Div:
    """Checklist element for option selection.

    Args:
        label: The title that goes above the checklist.
        id: A unique selector for this element.
        options: A list of dictionaries of labels and values.
        values: A list of values that should be preselected in the checklist.
        inline: Whether the options of the checklist are displayed beside or below each other.
    """
    return html.Div(
        className="checklist-wrapper",
        children=[
            html.Label(label),
            dcc.Checklist(
                id=id,
                className=f"checklist{' checklist--inline' if inline else ''}",
                inline=inline,
                options=options,
                value=values,
            ),
        ],
    )


def radio(label: str, id: str, options: list, value: int, inline: bool = True) -> html.Div:
    """Radio element for option selection.

    Args:
        label: The title that goes above the radio.
        id: A unique selector for this element.
        options: A list of dictionaries of labels and values.
        value: The value of the radio that should be preselected.
        inline: Whether the options are displayed beside or below each other.
    """
    return html.Div(
        className="radio-wrapper",
        children=[
            html.Label(label),
            dcc.RadioItems(
                id=id,
                className=f"radio{' radio--inline' if inline else ''}",
                inline=inline,
                options=options,
                value=value,
            ),
        ],
    )


def generate_options(options_list: list) -> list[dict]:
    """Generates options for dropdowns, checklists, radios, etc."""
    return [{"label": label, "value": i} for i, label in enumerate(options_list)]


def generate_train_tab() -> html.Div:
    """Settings for training the model.

    Returns:
        html.Div: A Div containing the settings for latents and save file name.
    """
    qpu_options = [{"label": qpu, "value": qpu} for qpu in SOLVERS]

    return html.Div(
        className="settings",
        children=[
            dropdown(
                "QPU",
                "qpu-setting",
                qpu_options,
                value=DEFAULT_QPU if DEFAULT_QPU in SOLVERS else SOLVERS[0],
            ),
            slider(
                "Latents",
                "n-latents",
                SLIDER_LATENTS,
            ),
            slider(
                "Epochs",
                "n-epochs",
                SLIDER_EPOCHS,
            ),
            html.Label("Save to File Name (optional)"),
            dcc.Input(
                id="file-name",
                type="text",
            ),
        ],
    )


def generate_generate_tab() -> html.Div:
    """Settings for generating.

    Returns:
        html.Div: A Div containing the settings for selecting the training file and other settings.
    """

    return html.Div(
        className="settings",
        children=[
            dropdown(
                "Trained Model",
                "model-file-name",
                generate_options(["No Models Found (please train and save a model)"])
            ),
            checklist(
                "",
                "tune-params",
                generate_options(["Tune Parameters"]),
                [],
            ),
            html.Div([
                slider(
                    "Epochs",
                    "n-epochs-tune",
                    SLIDER_EPOCHS,
                ),
                html.Label("Noise (optional)", className="display-none"),
                dcc.Input(
                    id="noise",
                    type="number",
                    className="display-none",
                    **NOISE,
                ),
            ], id="tune-parameter-settings")
        ],
    )


def generate_settings_form() -> dcc.Tabs:
    """This function generates settings training and generating.

    Returns:
        dcc.Tabs: Tabs containing settings for training and generation.
    """
    return dcc.Tabs(
        id="setting-tabs",
        value="generate-tab",
        mobile_breakpoint=0,
        children=[
            dcc.Tab(
                label="Train",
                id="train-tab",
                className="tab",
                children=[
                    generate_train_tab(),
                    html.Div([
                        generate_run_buttons("Train", "Cancel Training"),
                        html.Div([
                            html.Progress(value="0", id={"type": "progress", "index": 0}),
                            html.Div([
                                html.P("Epochs Completed:", id={"type": "progress-caption-epoch", "index": 0}),
                                html.P("Batch:", id={"type": "progress-caption-batch", "index": 0}),
                            ], className="display-flex")
                        ], id={"type": "progress-wrapper", "index": 0}, className="visibility-hidden")
                    ]),
                ],
            ),
            dcc.Tab(
                label="Generate",
                id="generate-tab",
                value="generate-tab",
                className="tab",
                children=[
                    generate_generate_tab(),
                    html.Div([
                        generate_run_buttons("Generate", "Cancel Generation"),
                        html.Div([
                            html.Progress(value="0", id={"type": "progress", "index": 1}),
                            html.Div([
                                html.P("Epochs Completed:", id={"type": "progress-caption-epoch", "index": 1}),
                                html.P("Batch:", id={"type": "progress-caption-batch", "index": 1}),
                            ], className="display-flex")
                        ], id={"type": "progress-wrapper", "index": 1}, className="visibility-hidden")
                    ]),
                ],
            ),
        ],
    )


def generate_run_buttons(run_text: str, cancel_text: str) -> html.Div:
    """Run and cancel buttons to run the problem."""
    return html.Div(
        className="button-group",
        children=[
            html.Button(
                id=f'{"-".join(run_text.lower().split(" "))}-button',
                children=run_text,
                n_clicks=0,
                disabled=False,
            ),
            html.Button(
                id=f'{"-".join(cancel_text.lower().split(" "))}-button',
                children=cancel_text,
                n_clicks=0,
                className="display-none",
            ),
        ],
    )


def generate_problem_details_table_rows(solver: str, time_limit: int) -> list[html.Tr]:
    """Generates table rows for the problem details table.

    Args:
        solver: The solver used for optimization.
        time_limit: The solver time limit.

    Returns:
        list[html.Tr]: List of rows for the problem details table.
    """

    table_rows = (
        ("Solver:", solver, "Time Limit:", f"{time_limit}s"),
        ### Add more table rows here. Each tuple is a row in the table.
    )

    return [html.Tr([html.Td(cell) for cell in row]) for row in table_rows]


def problem_details(index: int) -> html.Div:
    """Generate the problem details section.

    Args:
        index: Unique element id to differentiate matching elements.
            Must be different from left column collapse button.

    Returns:
        html.Div: Div containing a collapsable table.
    """
    return html.Div(
        id={"type": "to-collapse-class", "index": index},
        className="details-collapse-wrapper collapsed",
        children=[
            # Problem details collapsible button and header
            html.Button(
                id={"type": "collapse-trigger", "index": index},
                className="details-collapse",
                children=[
                    html.H5("Problem Details"),
                    html.Div(className="collapse-arrow"),
                ],
            ),
            html.Div(
                className="details-to-collapse",
                children=[
                    html.Table(
                        className="solution-stats-table",
                        children=[
                            # Problem details table header (optional)
                            html.Thead(
                                [
                                    html.Tr(
                                        [
                                            html.Th(
                                                colSpan=2,
                                                children=["Problem Specifics"],
                                            ),
                                            html.Th(
                                                colSpan=2,
                                                children=["Run Time"],
                                            ),
                                        ]
                                    )
                                ]
                            ),
                            # A Dash callback function will generate content in Tbody
                            html.Tbody(id="problem-details"),
                        ],
                    ),
                ],
            ),
        ],
    )


def create_interface():
    """Set the application HTML."""
    return html.Div(
        id="app-container",
        children=[
            # Below are any temporary storage items, e.g., for sharing data between callbacks.
            dcc.Store(id="batch-size"),
            # Header brand banner
            html.Div(
                id="popup",
                className="display-none",
                children=[
                    html.Div([
                        html.H2("Inaccessible QPU"),
                        html.P("The model selected was trained on a QPU that you do not have access to."),
                        html.P("Please select or train a new model."),
                        html.P("x", id="popup-toggle")
                    ])
                ]
            ),
            html.Div(className="banner", children=[html.Img(src=THUMBNAIL)]),
            # Settings and results columns
            html.Div(
                className="columns-main",
                children=[
                    # Left column
                    html.Div(
                        id={"type": "to-collapse-class", "index": 0},
                        className="left-column",
                        children=[
                            html.Div(
                                className="left-column-layer-1",  # Fixed width Div to collapse
                                children=[
                                    html.Div(
                                        className="left-column-layer-2",  # Padding and content wrapper
                                        children=[
                                            html.Div(
                                                [
                                                    html.H1(MAIN_HEADER),
                                                    html.P(DESCRIPTION),
                                                ],
                                                className="header-wrapper",
                                            ),
                                            generate_settings_form(),
                                        ],
                                    )
                                ],
                            ),
                            # Left column collapse button
                            html.Div(
                                html.Button(
                                    id={"type": "collapse-trigger", "index": 0},
                                    className="left-column-collapse",
                                    children=[html.Div(className="collapse-arrow")],
                                ),
                            ),
                        ],
                    ),
                    # Right column
                    html.Div(
                        className="right-column",
                        children=[
                            dcc.Tabs(
                                id="tabs",
                                value="input-tab",
                                mobile_breakpoint=0,
                                children=[
                                    dcc.Tab(
                                        label="MNIST Training Data",
                                        id="input-tab",
                                        value="input-tab",  # used for switching tabs programatically
                                        className="tab",
                                        children=[
                                            html.Div(
                                                dcc.Graph(
                                                    figure=display_input_data(),
                                                    id="fig-input",
                                                    responsive=True,
                                                    config={
                                                        "displayModeBar": False,
                                                    },
                                                ),
                                                className="graph",
                                            ),
                                        ]
                                    ),
                                    dcc.Tab(
                                        label="Generated Images",
                                        id="results-tab",
                                        className="tab",
                                        disabled=True,
                                        children=[
                                            html.Div(
                                                className="tab-content-results",
                                                children=[
                                                    html.Div(
                                                        className="graph-wrapper-flex",
                                                        children=[
                                                            html.Div(
                                                                [
                                                                    html.H3("Generated Images"),
                                                                    html.Div(
                                                                        dcc.Graph(
                                                                            id="fig-output",
                                                                            responsive=True,
                                                                            config={
                                                                                "displayModeBar": False,
                                                                            },
                                                                        ),
                                                                        className="graph",
                                                                    )
                                                                ],
                                                            ),
                                                            html.Div(
                                                                [
                                                                    html.H3("Reconstructed Images Comparison"),
                                                                    html.Div(
                                                                        dcc.Graph(
                                                                            id="fig-reconstructed",
                                                                            responsive=True,
                                                                            config={
                                                                                "displayModeBar": False
                                                                            },
                                                                        ),
                                                                        className="graph",
                                                                    )
                                                                ],
                                                            ),
                                                        ],
                                                    ),
                                                ],
                                            )
                                        ],
                                    ),
                                    dcc.Tab(
                                        label="Loss Graphs",
                                        id="loss-tab",
                                        className="tab",
                                        disabled=True,
                                        children=[
                                            html.Div(
                                                className="tab-content-results",
                                                children=[
                                                    html.Div(
                                                        className="graph-wrapper",
                                                        children=[
                                                            html.Div(
                                                                dcc.Graph(
                                                                    id="fig-loss",
                                                                    responsive=True,
                                                                    config={
                                                                        "displayModeBar": False
                                                                    },
                                                                ),
                                                                className="graph",
                                                            ),
                                                        ],
                                                    ),
                                                ],
                                            )
                                        ],
                                    ),
                                ],
                            )
                        ],
                    ),
                ],
            ),
        ],
    )
