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

"""This file stores input parameters for the app."""

# THEME_COLOR is used for the button, text, and banner and should be dark
# and pass accessibility checks with white: https://webaim.org/resources/contrastchecker/
# THEME_COLOR_SECONDARY can be light or dark and is used for sliders, loading icon, and tabs
THEME_COLOR = "#074C91"  # D-Wave dark blue default #074C91
THEME_COLOR_SECONDARY = "#2A7DE1"  # D-Wave blue default #2A7DE1

THUMBNAIL = "static/dwave_logo.svg"

APP_TITLE = "ML Image Generation (MNIST)"
MAIN_HEADER = "ML Image Generation (MNIST)"
DESCRIPTION = """\
Machine Learning MNIST training and image generation using a Discrete Variational
Autoencoder (DVAE) and a Graph Restricted Boltzmann Machine (GRBM).
"""

QPU = "Advantage2_system3.1"

#######################################
# Sliders, buttons and option entries #
#######################################

SLIDER_LATENTS = {
    "min": 128,
    "max": 512,
    "step": 64,
    "value": 256,
}

SLIDER_EPOCHS = {
    "min": 1,
    "max": 60,
    "step": 1,
    "value": 10,
}

NOISE = {
    "min": 0,
    "max": 1,
    "step": 0.01,
}
