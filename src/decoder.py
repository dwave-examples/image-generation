import torch


class Decoder(torch.nn.Module):
    def __init__(self, n_latents: int = 256):
        super().__init__()
        ...

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pass
