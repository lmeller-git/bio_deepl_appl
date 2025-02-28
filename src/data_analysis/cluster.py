import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from src.utils import Plotter


class Clusterplotter(Plotter):
    def __init__(self):
        self.df = None

    def plot(self) -> None:
        sns.clustermap(self.df.pivot(columns="mutation", values="ddG_ML").T, figsize=(10, 5), cmap="viridis", annot=True)
        plt.show()

    def update(self, csv: str) -> None:
        self.df = pd.read_csv(csv)
        self.df = df[["mut_type", "ddG_ML"]]
        self.df[["mut_from", "mut_to"]] = df["mut_type"].str.extract(r"([A-Z])\d*([A-Z])")
        self.df["mutation"] = df["mut_from"] + df["mut_to"]
        self.df.drop(columns=["mut_from", "mut_to"], inplace=True)
        self.df.dropna(subset=["mutation"], inplace=True)
        

    def clear(self):
        self.df = None


def cluster_plot(p: str = ".data/project_data/mega_train.csv"):
    plot = Clusterplotter()
    plot.update(p)
    plot.plot()



