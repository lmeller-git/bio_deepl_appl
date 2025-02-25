from abc import ABC, abstractmethod
from collections import defaultdict
import matplotlib.pyplot as plt

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


class LossPlotter(Plotter):
    x: dict[list[float]]
    y: dict[list[float]]
    metric: str

    def __init__(self, metric: str = "rmse"):
        super().__init__()
        (self.x, self.y) = (defaultdict(list), defaultdict(list))
        self.metric = metric

    def plot(self) -> None:
        keys = []
        for k, x in self.x.items():
            y = self.y[k]
            plt.plot(x, y)
            keys.append(k)
        plt.legend(keys)
        plt.show()

    def update(self, key: str, y: float, x: float | None = None) -> None:
        self.x[key].append(x if x is not None else len(self.x[key]))
        self.y[key].append(y)

    def clear(self) -> None:
        self = LossPlotter(self.metric)

class EmptyPlotter(Plotter):

    def __init__(self):
        super().__init__()

    def plot(self) -> None:
        pass

    def update(self) -> None:
        pass

    def clear(self) -> None:
        pass
