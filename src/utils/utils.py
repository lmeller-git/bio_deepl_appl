import torch
from abc import ABC, abstractmethod


class TrainParams:
    pass


def save_model(model: torch.nn.Module, path: str = "./out/best_model.pth") -> None:
    torch.save(model.state_dict(), path)


def load_model(path: str = "./out/best_model.pth") -> torch.nn.Module:
    torch.load(path)


def save_params(params: TrainParams, path: str = "./out/params.csv") -> None:
    with open(path, "w") as f:
        f.write(f"{params.epochs},{params.batch_size},{params.lr}")
    pass


def weight_reset(m):
    reset_parameters = getattr(m, "reset_parameters", None)
    if callable(reset_parameters):
        m.reset_parameters()

    for child in m.children():
        weight_reset(child)


class Plotter(ABC):
    def __init__(self):
        super().__init__()
        pass

    @abstractmethod
    def plot(self) -> None:
        pass

    @abstractmethod
    def update(self, *args, **kwargs) -> None:
        pass

    @abstractmethod
    def clear(self) -> None:
        pass


class EmptyPlotter(Plotter):
    def __init__(self):
        super().__init__()

    def plot(self) -> None:
        pass

    def update(self) -> None:
        pass

    def clear(self) -> None:
        pass
