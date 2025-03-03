from collections import defaultdict
import matplotlib.pyplot as plt
from src.utils import Plotter


class LossPlotter(Plotter):
    x: dict[list[float]]
    y: dict[list[float]]
    metric: str

    def __init__(self, metric: str = "rmse", *args, **kwargs):
        super().__init__(*args, **kwargs)
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

    def __repr__(self):
        return self.metric + "_plot"
