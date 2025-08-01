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


class Decoder(torch.nn.Module):
    """A decoder network that maps latent variables to images."""

    def __init__(self, n_latents: int):
        super().__init__()
        channels = [n_latents, 128, 64, 32, 1]
        layers = []

        # The input will be of shape (batch_size, n_latents), we need to project it
        # to the shape (batch_size, n_latents, 2, 2)
        self.increase_latent_dim = torch.nn.Linear(n_latents, n_latents * 2 * 2)
        self.make_2x2_images = torch.nn.Unflatten(-1, (n_latents, 2, 2))
        self.merge_batch_dim_and_replica_dim = torch.nn.Flatten(start_dim=0, end_dim=1)

        for i in range(len(channels) - 1):
            # A transposed convolutional layer does not modify the image size
            layers.append(
                torch.nn.ConvTranspose2d(
                    channels[i], channels[i + 1], kernel_size=3, stride=1, padding=1
                )
            )

            # Batch normalisation is used to stabilise the learning process
            layers.append(torch.nn.BatchNorm2d(channels[i + 1]))
            layers.append(torch.nn.Dropout2d(0.2))
            # We upsample the image size by 2
            layers.append(torch.nn.Upsample(scale_factor=2))
            # Finally, we apply a non-linearity
            layers.append(torch.nn.LeakyReLU())

        # We append a last convolutional transpose layer to obtain the final image
        layers.append(
            torch.nn.ConvTranspose2d(channels[-1], channels[-1], kernel_size=3, stride=1, padding=1)
        )
        self.convtrans = torch.nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.increase_latent_dim(x)
        x = self.make_2x2_images(x)
        batch_dim = x.shape[0]
        replica_dim = x.shape[1]
        x = self.merge_batch_dim_and_replica_dim(x)
        x = self.convtrans(x)

        return x.reshape(batch_dim, replica_dim, *x.shape[1:])
