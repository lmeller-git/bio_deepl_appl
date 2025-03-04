import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from src.utils import Plotter


class Clusterplotter(Plotter):
    def __init__(self):
        self.df_pivot = None

    def plot(self) -> None:
        sns.clustermap(self.df_pivot, figsize=(10, 5), cmap="viridis", annot=True)
        plt.show()

    def update(self, csv: str) -> None:
        df = pd.read_csv(csv)
        df = df[["mut_type", "ddG_ML"]]
        df[["mut_from", "mut_to"]] = df["mut_type"].str.extract(r"([A-Z])\d*([A-Z])")
        df["mutation"] = df["mut_from"] + df["mut_to"]
        df.drop(columns=["mut_from", "mut_to"], inplace=True)
        df.dropna(subset=["mutation"], inplace=True)
        df_pivot = df.pivot(index="mutation", columns="mutation", values="ddG_ML")

    def clear(self):
        self.df_pivot = None


def cluster_plot(p: str = ".data/project_data/mega_train.csv"):
    plot = Clusterplotter()
    plot.update(p)
    plot.should_save("cluster_plot" + p.split("/")[-1])
    plot.plot()
