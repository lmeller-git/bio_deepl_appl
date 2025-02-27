from torch import nn
import torch


def block(
    in_shape: int, out: float, act: nn.Module = nn.ReLU, drp: float = 0.42
) -> nn.Module:
    layer = nn.Sequential(
        nn.Linear(in_shape, out, bias=False),
        act(),
        nn.BatchNorm1d(out),
        nn.Dropout(drp),
    )

    return layer


class BasicMLP(nn.Module):
    def __init__(self, in_shape: int = 768, hidden_dim: int = 128):
        super().__init__()
        self.input = block(in_shape, hidden_dim)
        self.hidden = block(hidden_dim, hidden_dim)
        self.out = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input(x)
        x = self.hidden(x)
        return self.out(x)


class MLP(nn.Module):
    def __init__(self, in_shape: int = 768, hidden_dim: tuple[int] = (128, 256, 128)):
        super().__init__()
        self.input = block(in_shape, hidden_dim[0])
        self.hidden = nn.Sequential(
            block(hidden_dim[0], hidden_dim[1]),
            block(hidden_dim[1], hidden_dim[2], drp=0.2),
        )
        self.out = nn.Linear(hidden_dim[2], 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input(x)
        x = self.hidden(x)
        return self.out(x)


class LeakyMLP(nn.Module):
    def __init__(self, in_shape: int = 768, hidden_dim: tuple[int] = (128, 256, 128)):
        super().__init__()
        self.input = block(in_shape, hidden_dim[0])
        self.hidden = nn.Sequential(
            block(hidden_dim[0], hidden_dim[1], act=nn.LeakyReLU),
            block(hidden_dim[1], hidden_dim[2], act=nn.LeakyReLU, drp=0.2),
        )
        self.out = nn.Linear(hidden_dim[2], 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input(x)
        x = self.hidden(x)
        return self.out(x)


class Siamese(nn.Module):
    def __init__(self, in_shape: int = 768, hidden_dim: int = 256, n_layers: int = 1):
        super().__init__()
        self.shared_layers = nn.ModuleList([block(in_shape, hidden_dim)])
        for _ in range(n_layers):
            self.shared_layers.append(block(hidden_dim, hidden_dim))
        self.output_dim = hidden_dim * 3
        self.head = nn.Sequential(
            block(self.output_dim, hidden_dim, drp=0.2), nn.Linear(hidden_dim, 1)
        )

    def forward_single(self, x):
        for layer in self.shared_layers:
            x = layer(x)
        return x

    def forward(self, wt, mut):
        wt = self.forward_single(wt)
        mut = self.forward_single(mut)
        diff = wt - mut
        combined = torch.cat([wt, mut, diff], dim=1)
        return self.head(combined)
