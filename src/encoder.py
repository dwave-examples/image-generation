# Copyright 2025 D-Wave
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

import torch


class Encoder(torch.nn.Module):
    """An encoder network that maps image data to latent spinstrings."""

    def __init__(self, n_latents: int):
        super().__init__()
        channels = [1, 32, 64, 128, n_latents]
        layers = []

        for i in range(len(channels) - 1):
            # A convolutional layer does not modify the image size
            layers.append(
                torch.nn.Conv2d(channels[i], channels[i + 1], kernel_size=3, stride=1, padding=1)
            )
            # Batch normalisation is used to stabilise the learning process
            layers.append(torch.nn.BatchNorm2d(channels[i + 1]))
            # We downsample the image size by 2
            layers.append(torch.nn.MaxPool2d(kernel_size=2, stride=2))
            # Finally, we apply a non-linearity
            layers.append(torch.nn.LeakyReLU())

        layers = layers[:-1]  # Remove the last LeakyReLU
        self.conv = torch.nn.Sequential(*layers)
        self.flatten_last_two_dims = torch.nn.Flatten(start_dim=-2, end_dim=-1)
        self.projection = torch.nn.Linear(2 * 2, 1)
        self.flatten = torch.nn.Flatten()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.flatten_last_two_dims(x)
        x = self.projection(x)

        return self.flatten(x)  # .clip(-2.8, 2.8)
