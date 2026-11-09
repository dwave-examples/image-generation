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
from unittest.mock import Mock, patch

import torch

import app
from demo_callbacks import generate, refresh_qpus
from src.generation import GenerationResult


class DemoTests(unittest.TestCase):
    def test_http_layout_and_callback_registration(self):
        client = app.server.test_client()
        for path in (
            "/",
            "/_dash-layout",
            "/_dash-dependencies",
            "/static/dwave_logo.svg",
        ):
            with self.subTest(path=path):
                response = client.get(path)
                self.assertEqual(response.status_code, 200)
                response.close()
        layout = client.get("/_dash-layout").get_json()
        components = []

        def walk(component):
            if isinstance(component, dict):
                if "props" in component:
                    components.append(component)
                for child in component.values():
                    walk(child)
            elif isinstance(component, list):
                for child in component:
                    walk(child)

        walk(layout)
        ids = {component["props"].get("id") for component in components}
        self.assertTrue({"qpu", "generate", "gallery"} <= ids)
        self.assertFalse(
            any(
                word in str(identifier).lower()
                for identifier in ids
                for word in ("train", "tune", "epoch", "upload")
            )
        )

    @patch("demo_callbacks.get_engine")
    def test_refresh_preserves_user_qpu_selection(self, get_engine):
        options = [
            {"label": name, "value": name} for name in ("Advantage2_system1", "other")
        ]
        get_engine.return_value.available_solvers.return_value = options, []
        self.assertEqual(refresh_qpus(1, "other")[1], "other")
        self.assertEqual(refresh_qpus(1, "removed")[1], "Advantage2_system1")

    @patch("demo_callbacks.get_engine")
    def test_initialization_failure_is_visible_not_fake_availability(self, get_engine):
        get_engine.side_effect = RuntimeError("Failed to load model")
        with self.assertLogs("demo_callbacks", level="ERROR"):
            options, selected, message = refresh_qpus(1, None)
        self.assertEqual(options, [])
        self.assertIsNone(selected)
        self.assertEqual(message.className, "error")

    @patch("demo_callbacks.get_engine")
    def test_generation_failure_clears_previous_images_and_notifies_user(
        self, get_engine
    ):
        get_engine.return_value.generate.side_effect = RuntimeError("QPU unavailable")
        with self.assertLogs("demo_callbacks", level="ERROR"):
            images, message = generate(1, "test")
        self.assertEqual(images, [])
        self.assertEqual(message.className, "error")

    @patch("demo_callbacks.get_engine")
    def test_dash_generation_request_returns_exactly_four_inline_images(
        self, get_engine
    ):
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
        client = app.server.test_client()
        client.get("/_dash-dependencies")
        response = client.post(
            "/_dash-update-component",
            json={
                "output": "..gallery.children...generation-status.children..",
                "outputs": [
                    {"id": "gallery", "property": "children"},
                    {"id": "generation-status", "property": "children"},
                ],
                "inputs": [{"id": "generate", "property": "n_clicks", "value": 1}],
                "state": [{"id": "qpu", "property": "value", "value": "test"}],
                "changedPropIds": ["generate.n_clicks"],
            },
        )
        self.assertEqual(response.status_code, 200)
        cards = response.get_json()["response"]["gallery"]["children"]
        self.assertEqual(len(cards), 4)
        for card in cards:
            image = card["props"]["children"][0]
            self.assertTrue(image["props"]["src"].startswith("data:image/png;base64,"))
