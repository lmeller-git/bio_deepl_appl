import os

import numpy as np

import pandas as pd

import scipy

import sklearn.metrics as skmetrics


# plotting

import matplotlib.pyplot as plt

import seaborn as sns


# Pytorch

import torch


from torch.utils.data import DataLoader, Dataset

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

        tensor = torch.load(tensor_path)["mean_representations"][6]

        # wildtype embedding, uncomment if you want to use this, too
        try:
            tensor_path_wt = os.path.join(
                self.tensor_folder, self.wt_names[idx] + ".pt"
            )
            tensor_wt = torch.load(tensor_path_wt)["mean_representations"][6]
        except FileNotFoundError:
            tensor_path_wt = os.path.join(self.tensor_folder, self.ids[idx] + ".pt")
            tensor_wt = torch.load(tensor_path_wt)["mean_representations"][6]

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
        dataset_val, batch_size=batch_size, shuffle=False, num_workers=16
    )

    dataloader_test = DataLoader(dataset_test, batch_size=32, shuffle=False)

    return (dataloader_train, dataloader_val, dataloader_test)

    # your code
    preds = []

    all_y = []

    # save all predictions

    for batch in dataloader_val:
        # adjust this to work with your model

        x, y = batch

        y_hat = model(x)

        preds.append(y_hat.squeeze().detach().numpy())

        all_y.append(y.detach().numpy())

    # concatenate and plot

    preds = np.concatenate(preds)

    all_y = np.concatenate(all_y)

    sns.regplot(x=preds, y=all_y)

    plt.xlabel("Predicted ddG")

    plt.ylabel("Measured ddG")

    # get RMSE, Pearson and Spearman correlation

    print("RMSE:", skmetrics.mean_squared_error(all_y, preds, squared="False"))

    print("Pearson r:", scipy.stats.pearsonr(preds, all_y))

    print("Spearman r:", scipy.stats.spearmanr(preds, all_y))


if __name__ == "__main__":
    pass
