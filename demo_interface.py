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

"""An inference-only interface with a QPU selector and four generated faces."""

from dash import dcc, html

from demo_configs import APP_TITLE, CANDIDATE_COUNT, DESCRIPTION


def create_interface() -> html.Main:
    return html.Main(
        className="ffhq-demo",
        children=[
            html.Header(
                [
                    html.Img(
                        src="/static/dwave_logo.svg", alt="D-Wave", className="brand"
                    ),
                    html.H1(APP_TITLE),
                    html.P(DESCRIPTION),
                ]
            ),
            html.Section(
                className="controls",
                children=[
                    html.Label("QPU with valid precomputed embeddings", htmlFor="qpu"),
                    dcc.Dropdown(id="qpu", options=[], value=None, clearable=False),
                    html.Div(
                        className="actions",
                        children=[
                            html.Button("Refresh QPUs", id="refresh-qpus", n_clicks=0),
                            html.Button(
                                "Generate four faces",
                                id="generate",
                                n_clicks=0,
                                disabled=True,
                            ),
                        ],
                    ),
                    html.P(
                        f"Each request samples {CANDIDATE_COUNT} fresh candidates. "
                        "No model fitting or embedding search is performed."
                    ),
                    dcc.Loading(
                        html.Div(id="availability", role="status"), type="circle"
                    ),
                ],
            ),
            dcc.Loading(
                type="circle",
                children=html.Section(
                    [
                        html.Div(
                            id="generation-status",
                            role="status",
                            **{"aria-live": "polite"},
                        ),
                        html.Div(id="gallery", className="face-grid"),
                    ]
                ),
            ),
        ],
    )
