# Copyright 2026 D-Wave
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

"""Keep heavy inference objects in one process, not in spawned background callbacks."""

import logging
from threading import Lock

import dash
from dash import Input, Output, State, html

from demo_configs import DEFAULT_QPU
from src.generation import get_engine, image_data_uri

logger = logging.getLogger(__name__)
initialization_lock = Lock()


@dash.callback(
    Output("qpu", "options"),
    Output("qpu", "value"),
    Output("availability", "children"),
    Input("refresh-qpus", "n_clicks"),
    State("qpu", "value"),
    running=[(Output("refresh-qpus", "disabled"), True, False)],
)
def refresh_qpus(_clicks: int, current: str | None):
    try:
        with initialization_lock:
            options, unavailable = get_engine().available_solvers()
    except Exception:
        logger.exception(
            "Could not initialize FFHQ inference or discover supported QPUs."
        )
        return (
            [],
            None,
            html.P(
                "Could not load the inference models or query QPUs. See the server log for details, "
                "then refresh to retry.",
                className="error",
            ),
        )
    names = [option["value"] for option in options]
    selected = current if current in names else None
    if selected is None and names:
        selected = DEFAULT_QPU if DEFAULT_QPU in names else names[0]
    messages = [html.P(message) for message in unavailable]
    if not options:
        messages.insert(
            0,
            html.P("No available QPU has a valid saved embedding.", className="error"),
        )
    return options, selected, messages


@dash.callback(Output("generate", "disabled"), Input("qpu", "value"))
def generation_disabled(solver: str | None) -> bool:
    return solver is None


@dash.callback(
    Output("gallery", "children"),
    Output("generation-status", "children"),
    Input("generate", "n_clicks"),
    State("qpu", "value"),
    prevent_initial_call=True,
    running=[
        (Output("generate", "disabled", allow_duplicate=True), True, False),
        (Output("qpu", "disabled"), True, False),
        (Output("refresh-qpus", "disabled", allow_duplicate=True), True, False),
    ],
)
def generate(_clicks: int, solver: str | None):
    if solver is None:
        return [], html.P("Select an available QPU first.", className="error")
    try:
        result = get_engine().generate(solver)
        cards = [
            html.Figure(
                [
                    html.Img(
                        src=image_data_uri(image),
                        alt=f"Generated FFHQ face {index + 1}",
                    ),
                    html.Figcaption(
                        f"Quality rank {rank}/10 | eDifFIQA {float(score):.4f}"
                    ),
                ]
            )
            for index, (image, score, rank) in enumerate(
                zip(result.images, result.scores, result.quality_ranks)
            )
        ]
    except Exception:
        logger.exception("FFHQ image generation failed on %s.", solver)
        return [], html.P(
            "Generation failed; no cached or substitute samples were used. "
            "See the server log for details. Refresh QPUs if the working graph has changed.",
            className="error",
        )
    info = result.metadata
    return cards, (
        f"{info['solver']} | embedding {info['embedding_id']} | "
        f"{info['candidates']} fresh candidates | EMA step {info['checkpoint_step']:,} | "
        f"{info['seconds']:.1f} seconds"
    )
