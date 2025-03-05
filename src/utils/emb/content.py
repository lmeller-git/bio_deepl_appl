import os

import numpy as np

import pandas as pd

import scipy

import sklearn.metrics as skmetrics

import time

# plotting

import matplotlib.pyplot as plt

import seaborn as sns
from torch import nn

# Pytorch

import torch

import src.data_analysis.validation as d_validation
from torch.utils.data import DataLoader, Dataset
from src.utils.utils import Plotter

# the dataloaders load the tensors from memory one by one, could potentially become a bottleneck


class ProtEmbeddingDataset(Dataset):
    """

    Dataset for the embeddings of the mutated sequences

    You can the get_item() method to return the data in the format you want

    """

    def __init__(self, tensor_folder, csv_file, id_col="name", label_col="ddG_ML"):
        """

        Initialize the dataset

        input at init:

            tensor_folder: path to the directory with the embeddings we want to use, eg. "/home/data/mega_train_embeddings"

            cvs_file: path to the csv file corresponding to the data, eg. "home/data/mega_train.csv"

        """

        self.tensor_folder = tensor_folder

        self.df = pd.read_csv(csv_file, sep=",")

        # only use the mutation rows

        self.df = self.df[self.df.mut_type != "wt"]

        # get the labels and ids

        self.labels = torch.tensor(self.df[label_col].values)

        self.ids = self.df[id_col].values

        self.wt_names = self.df["WT_name"].values

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        # load embeddings

        # mutation embedding

        tensor_path = os.path.join(self.tensor_folder, self.ids[idx] + ".pt")

        tensor = torch.load(tensor_path, weights_only=True)["mean_representations"][6]

        # wildtype embedding, uncomment if you want to use this, too
        try:
            tensor_path_wt = os.path.join(
                self.tensor_folder, self.wt_names[idx] + ".pt"
            )
            tensor_wt = torch.load(tensor_path_wt, weights_only=True)[
                "mean_representations"
            ][6]
        except FileNotFoundError:
            tensor_path_wt = os.path.join(self.tensor_folder, self.ids[idx] + ".pt")
            tensor_wt = torch.load(tensor_path_wt, weights_only=True)[
                "mean_representations"
            ][6]

        label = self.labels[idx]  # ddG value
        # simple difference between embedings
        # TODO think of better ways (maybe Siamese network?)
        # tensor = tensor_wt - tensor
        # returns a tuple of the input embedding and the target ddG values

        return (tensor_wt, tensor), label.float()


# usage


# make sure to adjust the paths to where your files are located
def model():
    pass


def load_df(p: str, batch_size: int = 1024):
    dataset_train = ProtEmbeddingDataset(
        p + "project_data/mega_train_embeddings", p + "project_data/mega_train.csv"
    )

    dataset_val = ProtEmbeddingDataset(
        p + "project_data/mega_val_embeddings", p + "project_data/mega_val.csv"
    )

    dataset_test = ProtEmbeddingDataset(
        p + "project_data/mega_test_embeddings", p + "project_data/mega_test.csv"
    )

    dataloader_train = DataLoader(
        dataset_train, batch_size=batch_size, shuffle=True, num_workers=16
    )

    dataloader_val = DataLoader(
        dataset_val, batch_size=batch_size, shuffle=False, num_workers=0
    )

    dataloader_test = DataLoader(dataset_test, batch_size=32, shuffle=False)

    return (dataloader_train, dataloader_val, dataloader_test)


class PlotMutDist(Plotter):
    def plot(self):
        plt.figure(figsize=(14, 8))
        sns.barplot(
            data=self.all_counts,
            x="mutation",
            y="Occurrences",
            hue="Dataset",
            palette="viridis",
        )

        plt.xticks(rotation=90)  # Rotate mutation type labels for better readability
        plt.title("Occurrences of Mutation Types Across Datasets")
        plt.xlabel("Mutation Type")
        plt.ylabel("Occurrences")
        plt.legend(title="Dataset")
        plt.tight_layout()
        plt.show()
        pass

    def update(self, train: pd.DataFrame, val: pd.DataFrame, test: pd.DataFrame):
        train_counts = train["mutation"].value_counts().reset_index(name="Occurrences")
        train_counts["Dataset"] = "train"
        test_counts = test["mutation"].value_counts().reset_index(name="Occurrences")
        test_counts["Dataset"] = "test"
        val_counts = val["mutation"].value_counts().reset_index(name="Occurrences")
        val_counts["Dataset"] = "val"
        val_counts.columns = ["mutation", "Occurrences", "Dataset"]
        test_counts.columns = ["mutation", "Occurrences", "Dataset"]
        train_counts.columns = ["mutation", "Occurrences", "Dataset"]
        self.all_counts = pd.concat(
            [train_counts, val_counts, test_counts], ignore_index=True
        )
        # self.all_counts.rename(columns={"index": "Mutation Type"}, inplace=True)

    def clear(self):
        pass


def get_mut_type(df: pd.DataFrame) -> pd.DataFrame:
    df = df[["mut_type"]]
    df[["mut_from", "mut_to"]] = df["mut_type"].str.extract(r"([A-Z])\d*([A-Z])")
    df["mutation"] = df["mut_from"] + df["mut_to"]
    df.drop(columns=["mut_from", "mut_to"], inplace=True)
    df.dropna(inplace=True)
    return df


def plot_mut_dist(data: str = "./data/", which: list[str] = None, name: str = ""):
    test = pd.read_csv(data + "project_data/mega_test.csv")
    val = pd.read_csv(data + "project_data/mega_val.csv")
    train = pd.read_csv(data + "project_data/mega_train.csv")
    test = get_mut_type(test)
    train = get_mut_type(train)
    val = get_mut_type(val)
    if which is not None:
        train = train[train["mutation"].isin(which)]
        val = val[val["mutation"].isin(which)]
        test = test[test["mutation"].isin(which)]

    plotter = PlotMutDist()
    plotter.update(train, val, test)
    plotter.should_save("mut_dist_" + name)
    plotter.plot()


class PlotPredQualityMap(Plotter):
    def update(self, df: pd.DataFrame):
        self.df = df

    def plot(self):
        sns.clustermap(self.df, cmap="viridis", col_cluster=False)
        plt.show()

    def clear(self):
        self.df = None


class PlotPredQuality(Plotter):
    def update(self, low: pd.DataFrame, hi: pd.DataFrame, counts: pd.DataFrame):
        self.low = low
        self.hi = hi
        self.counts = counts

    def plot(self):
        fig, axes = plt.subplots(1, 3, figsize=(30, 6))

        sns.barplot(
            data=self.hi,
            x="Category",
            y="Value",
            hue="Group",
            palette="viridis",
            ax=axes[0],
        )
        axes[0].set_xlabel("Category")
        axes[0].set_ylabel("Value")
        axes[0].set_title("Top 10 Groups by Pearson Correlation")
        axes[0].legend(title="Group")

        sns.barplot(
            data=self.low,
            x="Category",
            y="Value",
            hue="Group",
            palette="magma",
            ax=axes[1],
        )
        axes[1].set_xlabel("Category")
        axes[1].set_ylabel("Value")
        axes[1].set_title("Bottom 10 Groups by Pearson Correlation")
        axes[1].legend(title="Group")

        sns.barplot(
            data=self.counts,
            x="mutation",
            y="Occurrences",
            hue="mutation",
            palette="crest",
            ax=axes[2],
        )
        axes[2].set_xlabel("Mutation")
        axes[2].set_ylabel("Occurrences")
        axes[2].set_title("Occurrences of Shown Groups")
        axes[2].tick_params(axis="x", rotation=45)

        plt.tight_layout()
        plt.show()

    def clear(self):
        self.low = None
        self.hi = None
        self.counts = None


def cross_validate(
    model: nn.Module, val: DataLoader, p: str = "./data/project_data/mega_val.csv"
):
    model.cpu().eval()
    # your code
    preds = []

    all_y = []

    # DO NOT REMOVE THIS PRINT STATEMENT (data race/deadlock or sth like that if n_workers > 0 in dataloader)
    print("")
    # time.sleep(1)
    # print("")

    # save all predictions
    for (wt, mut), y in val:
        # adjust this to work with your model
        wt = wt.cpu()
        mut = mut.cpu()
        y_hat = model(wt, mut)

        preds.append(y_hat.squeeze().detach().numpy())

        all_y.append(y.detach().numpy())

    # concatenate and plot
    # assuming data is not shuffled:
    preds = np.concatenate(preds)

    all_y = np.concatenate(all_y)

    plotter = PlotPredQuality()

    df = pd.read_csv(p)
    df = df[["mut_type"]]
    df[["mut_from", "mut_to"]] = df["mut_type"].str.extract(r"([A-Z])\d*([A-Z])")
    df["mutation"] = df["mut_from"] + df["mut_to"]
    df.drop(columns=["mut_from", "mut_to"], inplace=True)
    df.dropna(subset=["mutation"], inplace=True)
    df.dropna(inplace=True)
    df["y"] = all_y
    df["pred"] = preds

    groups = set(df["mutation"])

    group_data = []

    for group in groups:
        if len(df.loc[df["mutation"] == group]) < 2:
            continue
        res = d_validation.validate(
            df.loc[df["mutation"] == group, "y"],
            df.loc[df["mutation"] == group, "pred"],
            performance_metric=["rmse", "pearson", "spearman"],
            visualize=False,
        )
        group_data.append((group, res))

    print(len(group_data))

    group_data.sort(key=lambda x: x[1]["Pearson Correlation"], reverse=True)

    df_h = pd.concat(
        [
            pd.DataFrame(d.items(), columns=["Category", "Value"]).assign(Group=n)
            for n, d in group_data[:10]
        ]
    )

    df_l = pd.concat(
        [
            pd.DataFrame(d.items(), columns=["Category", "Value"]).assign(Group=n)
            for n, d in group_data[-10:]
        ]
    )

    group_counts = df["mutation"].value_counts().reset_index()
    group_counts.columns = ["mutation", "Occurrences"]

    selected_groups = list(df_h["Group"].unique()) + list(df_l["Group"].unique())
    group_counts_smol = group_counts[group_counts["mutation"].isin(selected_groups)]
    plotter.update(df_l, df_h, group_counts_smol)
    plotter.should_save("hi_lo_qual" + p.split("/")[-1])
    plotter.plot()

    pearson_dict = dict(group_data[:10] + group_data[-10:])

    df_heatmap = pd.DataFrame(
        {
            "Mutation": list(pearson_dict.keys()),
            "Pearson Correlation": [
                v["Pearson Correlation"] for k, v in pearson_dict.items()
            ],
        }
    )
    df_heatmap = pd.merge(
        df_heatmap, group_counts, left_on="Mutation", right_on="mutation", how="left"
    ).drop(columns=["mutation"])
    print(df_heatmap)

    df_heatmap["Occurrences"] = np.log10(df_heatmap["Occurrences"])

    # print(df_heatmap)

    df_heatmap.set_index("Mutation", inplace=True)

    map_plotter = PlotPredQualityMap()
    map_plotter.update(df_heatmap)
    map_plotter.should_save("qual_map" + p.split("/")[-1])
    map_plotter.plot()

    p = p.split("/")[:-2]
    s = ""
    for s_ in p:
        s += s_
    s += "/"
    plot_mut_dist(data=s, which=selected_groups, name=p.split("/")[-1])

    return

    sns.scatterplot(data=df, x="pred", y="y", hue="mutation", palette="viridis")

    plt.xlabel("Predicted ddG")

    plt.ylabel("Measured ddG")
    plt.show()


def validate(model: nn.Module, val: DataLoader):
    model.cpu()
    # your code
    preds = []

    all_y = []

    # save all predictions

    for (wt, mut), y in val:
        # adjust this to work with your model
        wt = wt.cpu()
        mut = mut.cpu()
        y_hat = model(wt, mut)

        preds.append(y_hat.squeeze().detach().numpy())

        all_y.append(y.detach().numpy())

    # concatenate and plot

    preds = np.concatenate(preds)

    all_y = np.concatenate(all_y)

    sns.regplot(x=preds, y=all_y)

    plt.xlabel("Predicted ddG")

    plt.ylabel("Measured ddG")
    plt.savefig(OUT + "pred_vs_lbl.png")
    if VERBOSITY >= 2:
        plt.show()
    else:
        plt.close()

    # get RMSE, Pearson and Spearman correlation

    print("RMSE:", skmetrics.mean_squared_error(all_y, preds))

    print("Pearson r:", scipy.stats.pearsonr(preds, all_y))

    print("Spearman r:", scipy.stats.spearmanr(preds, all_y))


if __name__ == "__main__":
    pass
