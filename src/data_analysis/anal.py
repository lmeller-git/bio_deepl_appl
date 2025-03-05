import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from src.utils import Plotter


class DistPlotter(Plotter):
    lbls: list[float]

    def __init__(self):
        super().__init__()
        self.lbls = None

    def plot(self):
        sns.histplot(data=self.lbls, x="ddG_ML", hue="set", palette="viridis")
        plt.show()

    def update(self, df: pd.DataFrame, *args, **kwargs):
        self.lbls = df

    def clear(self):
        self.lbls = None


def dist_plot(p: str = "./data/"):
    train = pd.read_csv(p + "project_data/mega_train.csv")
    val = pd.read_csv(p + "project_data/mega_val.csv")
    test = pd.read_csv(p + "project_data/mega_test.csv")
    train["set"] = "train"
    val["set"] = "val"
    test["set"] = "test"
    test = test[["set", "ddG_ML"]]
    train = train[["set", "ddG_ML"]]
    val = val[["set", "ddG_ML"]]
    f = pd.concat([train, val, test], ignore_index=True)
    plot = DistPlotter()
    plot.update(f)
    plot.should_save("dist_plot" + p.split("/")[-1])
    plot.plot()
