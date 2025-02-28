import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import umap
from src.utils import Plotter

class UMAP_Plotter(Plotter):
    
    def __init__(self):
        self.df = None
        
    def plot(self, labels: list = []):
        plt.figure(figsize=(10, 7))
        
        if labels:
            self.df["Label"] = labels

            scatter = sns.scatterplot(
                data=df,
                x="UMAP1", y="UMAP2",
                hue="Label",
                palette="tab10",
                edgecolor="black",
                alpha=0.8
            )
            plt.legend(title="Labels", bbox_to_anchor=(1.05, 1), loc='upper left')
        else:
            plt.scatter(df["UMAP1"], df["UMAP2"], color="blue", alpha=0.7)

        plt.title("UMAP of Δ Embeddings (Mutant - Wildtype)")
        plt.xlabel("UMAP Dimension 1")
        plt.ylabel("UMAP Dimension 2")
        plt.show()
        
    def update(self, wt_embeddings, mut_embeddings):
        delta_embeddings = mut_embeddings - wt_embeddings
        
        reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, metric="euclidean")
        umap_results = reducer.fit_transform(delta_embeddings)
        
        self.df = pd.DataFrame({
            "UMAP1": umap_results[:, 0],
            "UMAP2": umap_results[:, 1]
        })
        
        
    def clear(self):
        self.df = None


def umap_plot(wt_embeddings, mut_embeddings, lbls: list = []):
    plot = UMAP_Plotter()
    plot.update(wt_embeddings, mut_embeddings)
    plot.plot(lbls)