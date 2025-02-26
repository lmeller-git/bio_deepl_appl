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
            nn.Linear(in_shape, out, bias=False),
            nn.Sigmoid(),
            nn.BatchNorm1d(out),
            nn.Dropout(),
        )

        return layer

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input(x)
        x = self.hidden(x)
        return self.out(x)


class MLP(nn.Module):
    def __init__(self, in_shape: int):
        super().__init__()
        self.input = self.block(in_shape, 128)
        self.hidden = nn.Sequential(self.block(128, 256), self.block(256, 128))
        self.out = self.block(128, 1, act=nn.Sigmoid)

    def block(
        self, in_shape: int, out: float, act: nn.Module = nn.LeakyReLU
    ) -> nn.Module:
        layer = nn.Sequential(
            nn.Linear(in_shape, out, bias=False),
            act(),
            nn.BatchNorm1d(out),
            nn.Dropout(),
        )

        return layer

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input(x)
        x = self.hidden(x)
        return self.out(x)
