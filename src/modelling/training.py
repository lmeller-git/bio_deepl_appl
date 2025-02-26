import torch
from torch import nn
from sklearn.model_selection import KFold
from tqdm import tqdm
from torch.utils.data import DataLoader, sampler
import numpy as np

# TODO fix circular import
from src import data_analysis
from src.modelling.models import BasicMLP, MLP
from src.modelling.eval import LossPlotter
from src.utils import (
    load_df,
    Plotter,
    save_model,
    ProtEmbeddingDataset,
    EmptyPlotter,
    weight_reset,
)

global DEVICE
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.set_default_dtype(torch.float64)
torch.manual_seed(42)

"""
class CVDefinition:
    lr_split: float
    epoch_split: int
    batch_split: int

    def __init__(
        self, lr_split: float = 0.0, epoch_split: int = 0, batch_split: int = 0
    ):
        self.lr_split = lr_split
        self.epoch_split = epoch_split
        self.batch_split = batch_split
"""


class TrainParams:
    epochs: int
    train_df: str
    lr: float
    batch_size: int
    cv: int
    kfold_args: dict
    data: str
    # cv_def: CVDefinition

    def __init__(
        self,
        train_df,
        epochs: int = 10,
        lr: float = 1e-4,
        batch_size: int = 1024,
        cv: int = 0,
        # cv_definition: CVDefinition = CVDefinition(),
        data: str = "./data/",
        **kwargs,
    ):
        self.epochs = epochs
        self.lr = lr
        self.train_df = train_df
        self.batch_size = batch_size
        self.cv = cv
        self.kfold_args = kwargs
        # self.cv_def = cv_definition


class RMSELoss(nn.Module):
    def __init__(self, eps=1e-6):
        super().__init__()
        self.mse = nn.MSELoss()
        self.eps = eps

    def forward(self, yhat, y):
        loss = torch.sqrt(self.mse(yhat, y) + self.eps)
        return loss


def train(model: nn.Module | None, params: TrainParams):
    if params.cv > 0:
        kfold(params)
        return
    train_df, val_df, test_df = load_df(params.train_df, params.batch_size)

    plotter = LossPlotter()
    # model = BasicMLP(768)
    print("training model")
    train_loop(model, train_df, val_df, params, plotter)
    plotter.plot()
    save_model(model)
    print("model trained and saved")


def train_loop(
    model: nn.Module,
    train: DataLoader,
    test: DataLoader,
    params: TrainParams,
    plotter: Plotter = EmptyPlotter(),
):
    optim = torch.optim.Adam(model.parameters(), params.lr)
    criterion = RMSELoss(0)
    model.to(DEVICE)
    for epoch in tqdm(range(params.epochs)):
        print("step")
        model.train()
        step(model, optim, criterion, train, plotter)
        model.eval()
        with torch.no_grad():
            print(f"\nEpoch {epoch}:")
            validate(model, criterion, test, plotter)


def kfold(params: TrainParams) -> nn.Module:
    # TODO: Hyperparameter tuning via Vec<TrainParams>
    print("performing kfold cv")
    train_data = ProtEmbeddingDataset(
        params.train_df + "project_data/mega_train_embeddings",
        params.train_df + "project_data/mega_train.csv",
    )
    _val_data = ProtEmbeddingDataset(
        params.train_df + "project_data/mega_val_embeddings",
        params.train_df + "project_data/mega_val.csv",
    )

    kf = KFold(params.cv)
    models = [BasicMLP(768), MLP(768)]
    val_df = np.zeros((len(models), params.cv))
    train_df = np.zeros((len(models), params.cv))
    plotter = LossPlotter()
    kfold_plotter = LossPlotter("rmse kfold")
    kfold_params = []
    for i in range(params.cv):
        base_lr = params.lr
        base_epochs = params.epochs
        base_batch_size = 1024
        for k, v in params.kfold_args.items():
            match k:
                case "d_lr":
                    base_lr += i * v
                case "d_epoch":
                    base_epochs += i * v
                case "d_batch_size":
                    base_batch_size += i * v
                case _:
                    print(f"case {k} is not defined")
        kfold_params.append(
            TrainParams(params.train_df, base_epochs, base_lr, base_batch_size)
        )

    for split, (train_idc, val_idc) in tqdm(enumerate(kf.split(train_data.ids))):
        # print("\n")
        # print(split, train_idc, val_idc)
        train_sampler = sampler.SubsetRandomSampler(train_idc)
        val_sampler = sampler.SubsetRandomSampler(val_idc)
        train_split = DataLoader(
            train_data,
            batch_size=kfold_params[split].batch_size,
            shuffle=False,
            num_workers=16,
            sampler=train_sampler,
        )
        val_split = DataLoader(
            train_data,
            batch_size=kfold_params[split].batch_size,
            shuffle=False,
            num_workers=16,
            sampler=val_sampler,
        )
        for m, model in enumerate(models):
            train_loop(model, train_split, val_split, kfold_params[split], plotter)
            val_df[m, split] = sum(plotter.y["val loss"]) / len(plotter.y["val loss"])
            train_df[m, split] = sum(plotter.y["train loss"]) / len(
                plotter.y["train loss"]
            )
            plotter.clear()
            kfold_plotter.update("val model " + str(m), val_df[m, split], split)
            kfold_plotter.update("train model " + str(m), train_df[m, split], split)
            weight_reset(model)

    kfold_plotter.plot()
    print(train_df, val_df)
    (train_df, train_std) = (np.mean(train_df, axis=1), np.std(train_df, axis=1))
    (val_df, val_std) = (np.mean(val_df, axis=1), np.std(val_df, axis=1))
    print(train_df, val_df)
    best_model = np.argmin(train_df)
    print(
        f"Best model in fold {best_model}: {models[best_model]}\ntrain loss: {train_df[best_model]} | train std: {train_std[best_model]} | val loss: {val_df[best_model]} | val std: {val_std[best_model]}"
    )
    train(models[best_model], kfold_params[best_model])


def validate(model: nn.Module, criterion: nn.Module, df: DataLoader, plotter: Plotter):
    losses = []
    scc = []
    pcc = []
    for embs, lbl in df:
        (embs, lbl) = (embs.to(DEVICE), lbl.to(DEVICE))
        yhat = model(embs).squeeze()
        loss = criterion(yhat, lbl)
        losses.append(loss)
        r = data_analysis.validate(lbl, yhat, ["pearson", "spearman"], False)
        scc.append(r["Spearman Correlation"])
        pcc.append(r["Pearson Correlation"])

    loss = sum(losses) / len(losses)
    scc = sum(scc) / len(scc)
    pcc = sum(pcc) / len(pcc)
    plotter.update("val loss", loss.detach().cpu().numpy())
    print(f"\tLoss: {loss:.3f} | scc: {scc:.3f} | pcc: {pcc:.3f}")


def step(
    model: nn.Module,
    optim: torch.optim.Optimizer,
    criterion: nn.Module(),
    df: DataLoader,
    plotter: Plotter,
):
    losses = []
    for embs, lbl in df:
        (embs, lbl) = (embs.to(DEVICE), lbl.to(DEVICE))
        optim.zero_grad()
        criterion.zero_grad()
        yhat = model(embs).squeeze()
        loss = criterion(yhat, lbl)
        loss.backward()
        optim.step()
        losses.append(loss)
    losses = sum(losses) / len(losses)
    plotter.update("train loss", losses.detach().cpu().numpy())
