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

import unittest
from unittest.mock import patch

import torch
from dash.exceptions import PreventUpdate

from demo_configs import FFHQ_MODEL_NAME
from src.generation import GenerationResult

# Importing the original interface initializes solver choices. Keep unit tests offline.
with patch("src.generation.get_engine") as engine:
    engine.return_value.available_solvers.return_value = (
        [{"label": "test", "value": "test"}],
        [],
    )
    with patch("sys.argv", ["app.py"]):
        import app

from demo_callbacks import generate, initialize_training_model, train

app.app.layout = app.create_interface()


class DemoTests(unittest.TestCase):
    def test_original_layout_is_preserved_and_training_controls_are_disabled(self):
        client = app.app.server.test_client()
        for path in ("/", "/_dash-layout", "/_dash-dependencies"):
            self.assertEqual(client.get(path).status_code, 200)
        components = {}

        def walk(component):
            if isinstance(component, dict):
                props = component.get("props", {})
                if isinstance(props.get("id"), str):
                    components[props["id"]] = props
                for child in component.values():
                    walk(child)
            elif isinstance(component, list):
                for child in component:
                    walk(child)

        walk(client.get("/_dash-layout").get_json())
        self.assertTrue(
            {"setting-tabs", "fig-output", "fig-qpu-graph", "inference-qpu"}
            <= components.keys()
        )
        self.assertTrue(components["train-tab"]["disabled"])
        self.assertTrue(components["tune-params"]["options"][0]["disabled"])
        self.assertNotIn("data", components["example-image"])
        self.assertEqual(initialize_training_model(None)[1], FFHQ_MODEL_NAME)

    @patch("demo_callbacks.ModelWrapper")
    def test_training_is_also_disabled_server_side(self, model_wrapper):
        with self.assertRaises(PreventUpdate):
            train(None, 1, "test", 256, 1, "unused", None)
        model_wrapper.assert_not_called()

    @patch("demo_callbacks.get_engine")
    def test_generation_failure_clears_previous_images_and_notifies_user(
        self, get_engine
    ):
        get_engine.return_value.generate.side_effect = RuntimeError("QPU unavailable")
        with self.assertLogs("demo_callbacks", level="ERROR"):
            result = generate(1, FFHQ_MODEL_NAME, "test")
        self.assertEqual(len(result.fig_generated.data), 0)
        self.assertIn("Generation failed", result.problem_details_table[0].children)

    @patch("demo_callbacks.get_engine")
    def test_dash_generation_uses_existing_graph_with_four_image_grid(self, get_engine):
        get_engine.return_value.generate.return_value = GenerationResult(
            images=torch.zeros(4, 3, 2, 2),
            scores=torch.tensor([0.9, 0.8, 0.7, 0.6]),
            quality_ranks=[1, 3, 5, 10],
            generated_indices=[0, 4, 8, 12],
            metadata={
                "solver": "test",
                "embedding_id": 1,
                "candidates": 32,
                "checkpoint_step": 540000,
                "seconds": 1.2,
            },
        )
        client = app.app.server.test_client()
        client.get("/_dash-dependencies")
        key, callback = next(
            (key, callback)
            for key, callback in app.app.callback_map.items()
            if callback["inputs"] == [{"id": "generate-button", "property": "n_clicks"}]
        )
        outputs = [
            {"id": output.component_id, "property": output.component_property}
            for output in callback["output"]
        ]
        response = client.post(
            "/_dash-update-component",
            json={
                "output": key,
                "outputs": outputs,
                "inputs": [
                    {"id": "generate-button", "property": "n_clicks", "value": 1}
                ],
                "state": [
                    {
                        "id": "model-file-name",
                        "property": "value",
                        "value": FFHQ_MODEL_NAME,
                    },
                    {"id": "inference-qpu", "property": "value", "value": "test"},
                ],
                "changedPropIds": ["generate-button.n_clicks"],
            },
        )
        self.assertEqual(response.status_code, 200)
        result = response.get_json()["response"]
        source = result["fig-output"]["figure"]["data"][0]["source"]
        self.assertTrue(source.startswith("data:image/png;base64,"))
        self.assertFalse(result["results-tab"]["disabled"])
        self.assertEqual(result["tabs"]["value"], "results-tab")
        get_engine.return_value.generate.assert_called_once_with("test")
