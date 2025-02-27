from torch import nn
import torch


def block(in_shape: int, out: float, act: nn.Module = nn.ReLU) -> nn.Module:
    layer = nn.Sequential(
        nn.Linear(in_shape, out, bias=False),
        act(),
        nn.BatchNorm1d(out),
        nn.Dropout(0.42),
    )

    return layer


class BasicMLP(nn.Module):
    def __init__(self, in_shape: int, hidden_dim: int = 128):
        super().__init__()
        self.input = block(in_shape, hidden_dim)
        self.hidden = block(hidden_dim, hidden_dim)
        self.out = block(hidden_dim, 1, act=nn.Identity)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input(x)
        x = self.hidden(x)
        return self.out(x)


class MLP(nn.Module):
    def __init__(self, in_shape: int, hidden_dim: tuple[int] = (128, 256, 128)):
        super().__init__()
        self.input = block(in_shape, hidden_dim[0])
        self.hidden = nn.Sequential(
            block(hidden_dim[0], hidden_dim[1]), block(hidden_dim[1], hidden_dim[2])
        )
        self.out = block(128, 1, act=nn.Identity)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input(x)
        x = self.hidden(x)
        return self.out(x)


class LeakyMLP(nn.Module):
    def __init__(self, in_shape: int, hidden_dim: tuple[int] = (128, 256, 128)):
        super().__init__()
        self.input = block(in_shape, hidden_dim[0])
        self.hidden = nn.Sequential(
            block(hidden_dim[0], hidden_dim[1], act=nn.LeakyReLU),
            block(hidden_dim[1], hidden_dim[2], act=nn.LeakyReLU),
        )
        self.out = block(128, 1, act=nn.Identity)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input(x)
        x = self.hidden(x)
        return self.out(x)
