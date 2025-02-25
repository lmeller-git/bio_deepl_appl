from torch import nn
import torch


class BasicMLP(nn.Module):
    def __init__(self, in_shape: int):
        super().__init__()
        self.input = self.block(in_shape, 100)
        self.hidden = self.block(100, 100)
        self.out = self.block(100, 1)

    def block(self, in_shape: int, out: float) -> nn.Module:
        layer = nn.Sequential(
            nn.Linear(in_shape, out), nn.ReLU(), nn.BatchNorm1d(out), nn.Dropout()
        )

        return layer

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input(x)
        x = self.hidden(x)
        return self.out(x)
