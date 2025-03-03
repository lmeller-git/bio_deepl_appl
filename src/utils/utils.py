import torch
from abc import ABC, abstractmethod
import matplotlib.pyplot as plt

# from src.config import VERBOSITY, OUT


class TrainParams:
    pass


def save_model(model: torch.nn.Module, path: str = OUT + "best_model.pth") -> None:
    torch.save(model.state_dict(), path)


def load_model(path: str = OUT + "best_model.pth") -> torch.nn.Module:
    torch.load(path)


def save_params(params: TrainParams, path: str = OUT + "params.csv") -> None:
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
    out: str

    def __init__(self, out: str = OUT, *args, **kwargs):
        super().__init__()
        self.out = out
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

    def should_save(self, p: str = "") -> None:

        original_show = plt.show

        def custom_show(*args, **kwargs):
            try:
                plt.savefig(self.out + str(self) + "_" + p + ".png")
            finally:
                plt.show = original_show
                if VERBOSITY >= 2:
                    plt.show(*args, **kwargs)
                else:
                    plt.close()

        plt.show = custom_show

    def __repr__(self):
        return "plot"


class EmptyPlotter(Plotter):
    def __init__(self):
        super().__init__()

    def plot(self) -> None:
        pass

    def update(self) -> None:
        pass

    def clear(self) -> None:
        pass
