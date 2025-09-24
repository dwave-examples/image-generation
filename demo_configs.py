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

APP_TITLE = "ML Image Generation"
MAIN_HEADER = "ML Image Generation"
DESCRIPTION = """\
Machine Learning MNIST training and image generation using a Discrete Variational
Autoencoder (DVAE) and a Graph Restricted Boltzmann Machine (GRBM).
"""

DEFAULT_QPU = "Advantage2_system1.6"

GENERATE_NEW_MODEL_DIAGRAM = True  # If True, runs will update the model diagram in the input tab.

# The index (in the MNIST dataset) of the image to use as an example in the UI
# The first couple images in indices 0-10 are [5, 0, 4, 1, 9, 2, 1, 3, 1, 4, 3]
# The demo may need to be restarted to see this change
EXAMPLE_IMAGE_INDEX = 0

GRAPH_COLORS = ["#FF7006", "#17BEBB"]  # First color is for -1 second is for +1

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

SHARPEN_OUTPUT = False
UPPER_THRESHOLD = 0.6
LOWER_THRESHOLD = 0.4
