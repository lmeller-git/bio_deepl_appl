import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from src.utils import Plotter


class DistPlotter(Plotter):
    lbls: list[float]

    def __init__(self):
        super().__init__()
        self.lbls = []

    def plot(self):
        sns.histplot(data=self.lbls)
        plt.plot()

    def update(self, lbl: float, *args, **kwargs):
        self.lbls.append(lbl)

    def clear(self):
        self.lbls = []


def dist_plot(p: str = "./data/project_data/mega_train.csv"):
    f = pd.read_csv(p)
    lbls = f["ddG_ML"]
    plot = DistPlotter()
    for lbl in lbls:
        plot.update(lbl)

    plot.plot()
