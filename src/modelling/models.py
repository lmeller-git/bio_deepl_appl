from torch import nn
import torch


def block(
    in_shape: int, out: float, act: nn.Module = nn.ReLU, drp: float = 0.42
) -> nn.Module:
    layer = nn.Sequential(
        nn.Linear(in_shape, out),
        act(),
        nn.BatchNorm1d(out),
        nn.Dropout(drp),
    )

    return layer


class BasicMLP(nn.Module):
    def __init__(
        self,
        in_shape: int = 768,
        hidden_dim: int = 128,
        out_shape: int = 1,
        act: nn.Module = nn.ReLU,
    ):
        super().__init__()
        self.input = block(in_shape, hidden_dim, act=act)
        self.hidden = block(hidden_dim, hidden_dim, act=act)
        self.out = nn.Linear(hidden_dim, out_shape)

    def forward(self, wt, mut, *args, **kwargs):
        x = mut - wt
        x = self.input(x)
        x = self.hidden(x)
        return self.out(x)


class MLP(nn.Module):
    def __init__(
        self,
        in_shape: int = 768,
        hidden_dim: tuple[int] = (128, 256, 128),
        out_shape: int = 1,
        act: nn.Module = nn.ReLU(),
    ):
        super().__init__()
        self.input = block(in_shape, hidden_dim[0], act)
        self.hidden = nn.Sequential(
            block(hidden_dim[0], hidden_dim[1], act),
            block(hidden_dim[1], hidden_dim[2], drp=0.2, act=act),
        )
        self.out = nn.Linear(hidden_dim[2], out_shape)

    def forward(self, wt, mut, *args, **kwargs):
        x = mut - wt
        x = self.input(x)
        x = self.hidden(x)
        return self.out(x)


class Siamese(nn.Module):
    def __init__(
        self,
        in_shape: int = 768,
        hidden_dim: int = 256,
        n_layers: int = 1,
        out_shape: int = 1,
        act: nn.Module = nn.ReLU,
    ):
        super().__init__()
        self.shared_layers = nn.ModuleList([block(in_shape, hidden_dim, act)])
        for _ in range(n_layers):
            self.shared_layers.append(block(hidden_dim, hidden_dim, act))
        self.output_dim = hidden_dim * 3
        self.head = nn.Sequential(
            block(self.output_dim, hidden_dim, drp=0.2, act=act),
            nn.Linear(hidden_dim, out_shape),
        )

    def forward_single(self, x):
        for layer in self.shared_layers:
            x = layer(x)
        return x

    def forward(self, wt, mut, *args, **kwargs):
        wt = self.forward_single(wt)
        mut = self.forward_single(mut)
        diff = mut - wt
        combined = torch.cat([wt, mut, diff], dim=1)
        return self.head(combined)


class ExtendedSiamese(nn.Module):
    def __init__(
        self,
        in_shape: int = 768,
        hidden_dim: int = 256,
        n_layers: int = 1,
        out_shape: int = 1,
        act: nn.Module = nn.ReLU,
    ):
        super().__init__()
        # TODO LA/CA?
        self.shared_layers = nn.ModuleList([block(in_shape, hidden_dim, act)])
        for _ in range(n_layers):
            self.shared_layers.append(block(hidden_dim, hidden_dim, act))

        self.shared_layers.append(
            block(hidden_dim, out_shape, act=nn.Identity, drp=0.0)
        )

    def forward(self, wt, mut, *args, **kwargs):
        for layer in self.shared_layers:
            wt = layer(wt)
            mut = layer(mut)

        ddg = mut - wt

        return ddg


# TODO
class SiameseSequenceReg(nn.Module):
    def __init__(
        self,
        in_shape: int = 768,
        hidden_dim: int = 768,
        n_layers: int = 1,
        act: nn.Module = nn.ReLU,
        out_shape: int = 1,
    ):
        super().__init__()
        self.layers = nn.ModuleList([block(in_shape, hidden_dim, act)])
        for _ in range(n_layers):
            self.layers.append(block(hidden_dim, hidden_dim, act))

    def forward(self, seq_wt, seq_mut, *args, **kwargs):
        pass


class MixedModel(nn.Module):
    def __init__(self, in_shape: int = 768, hidden_dim: int = 256, n_layers: int = 1):
        super().__init__()
        self.emb_reg = BasicMLP(
            hidden_dim=hidden_dim, out_shape=hidden_dim, act=nn.ReLU
        )
        self.seq_reg = SiameseSequenceReg()(
            hidden_dim=hidden_dim, out_shape=hidden_dim, act=nn.ReLU
        )
        self.head = nn.Sequential(
            block(hidden_dim, hidden_dim, nn.LeakyReLU), nn.Linear(hidden_dim, 1)
        )

    def forward(self, wt, mut, seq_wt, seq_mut, *args, **kwargs):
        emb = self.emb_reg(wt, mut)
        seq = self.seq_reg(seq_wt, seq_mut)
        x = torch.cat([emb, seq], dim=1)
        return self.head(x)
