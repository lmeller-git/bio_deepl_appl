import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from src.utils import Plotter

class Clusterplotter(Plotter): 
    def __init__(self):
        self.df_pivot = None
        
    def plot(self) -> None:
        sns.clustermap(df_pivot, figsize=(10, 5), cmap="viridis", annot=True)
        plt.show()
        
    def update(self, csv: str) -> None:
        df = pd.read_csv(csv)
        df = df[['mut_type'], ['ddG_ML']]
        df["mutation"] = df["mut_type"].str.extract(r"([A-Z])\d*([A-Z])")  
        df["mutation"] = df["mutation"].apply(lambda x: f"{x[0]}{x[1]}" if isinstance(x, tuple) else "WT")
        df_pivot = df.pivot(index='mut_type', columns=None, values='ddG_ML')

        
    def clear(self):
        self.df_pivot = None
        
        
        
def cluster_plot(p: str = ".data/project_data/mega_train.csv"):
    plot = Clusterplotter()
    plot.update(p)
    plot.plot()


