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

import torch.nn as nn

import torch.nn.functional as F

import torch.optim as optim

from torch.utils.data import DataLoader, Dataset

import lightning as L



import torchmetrics

from torchmetrics.regression import PearsonCorrCoef
aa_alphabet = 'ACDEFGHIKLMNPQRSTVWY' # amino acid alphabet

aa_to_int = {aa: i for i, aa in enumerate(aa_alphabet)} # mapping from amino acid to number



# function to one hot encode sequence

def one_hot_encode(sequence):

    # initialize a zero matrix of shape (len(sequence), len(amino_acids))

    one_hot = torch.zeros(len(sequence), len(aa_alphabet))

    for i, aa in enumerate(sequence):

        # set the column corresponding to the amino acid to 1

        one_hot[i].scatter_(0, torch.tensor([aa_to_int[aa]]), 1)

    return one_hot





# sequence data, comes already batched, so treat accordingly in dataloader (batch_size=1)

class SequenceData(Dataset):

    def __init__(self, csv_file, label_col="ddG_ML"):

        """

        Initializes the dataset. 

        input:

            csv_file: path to the relevant data file, eg. "/home/data/mega_train.csv"

        """

        self.df = pd.read_csv(csv_file, sep=",")

        self.label_col = label_col

        # only have mutation rows

        self.df = self.df[self.df.mut_type!="wt"]

        # process the mutation row

        self.df["mutation_pos"] = self.df["mut_type"].apply(lambda x: int(x[1:-1])-1) # make position start at zero

        self.df["mutation_to"] = self.df["mut_type"].apply(lambda x: aa_to_int[x[-1]]) # give numerical label to mutation



        # group by wild type

        self.df = self.df.groupby("WT_name").agg(list)

        # get wild type names

        self.wt_names = self.df.index.values

        # precompute one-hot encoding for faster training

        self.encoded_seqs = {}

        for wt_name in self.wt_names:

            # get the correct row

            mut_row = self.df.loc[wt_name]

            seq = mut_row["wt_seq"][0]

            self.encoded_seqs[wt_name] = one_hot_encode(seq)



    def __len__(self):

        return len(self.df)



    def __getitem__(self, idx):

        # get the wild type name

        wt_name = self.wt_names[idx]

        # get the correct row

        mut_row = self.df.loc[wt_name]

        # get the wt sequence in one hot encoding

        sequence_encoding = self.encoded_seqs[wt_name]



        # create mask and target tensors

        mask = torch.zeros((1, len(sequence_encoding),20)) # will be 1 where we have a measurement

        target = torch.zeros((1, len(sequence_encoding),20)) # ddg values

        # all mutations from df

        positions = torch.tensor(mut_row["mutation_pos"])

        amino_acids = torch.tensor(mut_row["mutation_to"])

        # get the labels

        labels = torch.tensor(mut_row[self.label_col])



        for i in range(len(sequence_encoding)):

            mask[0,i,amino_acids[positions==i]] = 1 # one where we have data

            target[0,i,amino_acids[positions==i]] = labels[positions==i] # fill with ddG values

        

        # returns encoded sequence, mask and target sequence 

        return {"sequence": sequence_encoding[None,:,:].float(), "mask": mask, "labels": target}
# usage

dataset_train = SequenceData('project_data/mega_train.csv')

dataset_val= SequenceData('project_data/mega_val.csv')

dataset_test = SequenceData('project_data/mega_test.csv')



# use batch_size=1 bc we treat each sequence as one batch

dataloader_val = DataLoader(dataset_val, batch_size=1, shuffle=False)

dataloader_train = DataLoader(dataset_train, batch_size=1, shuffle=True)

dataloader_test = DataLoader(dataset_test, batch_size=1, shuffle=False)
# your code 
preds =[]

all_y = []



for batch in dataloader_val:

    # read from batch

    x = batch["sequence"][0]

    mask = batch["mask"][0]

    target = batch["labels"][0]

    ## adjust to work with your model

    # predict

    prediction = model(x)

    preds.append(prediction[mask==1].flatten().detach().numpy()) # flatten to create one dimensional vector from 2D sequence

    all_y.append(target[mask==1].flatten().detach().numpy()) # flatten to create one dimensional vector from 2D sequence



# concatenate and plot

preds= np.concatenate(preds)

all_y = np.concatenate(all_y)



sns.regplot(x=preds,y=all_y)

plt.xlabel("Predicted ddG")

plt.ylabel("Measured ddG")



# get RMSE, Pearson and Spearman correlation 

print("RMSE:", skmetrics.mean_squared_error(all_y, preds, squared="False"))

print("Pearson r:", scipy.stats.pearsonr(preds, all_y))

print("Spearman r:", scipy.stats.spearmanr(preds, all_y))
