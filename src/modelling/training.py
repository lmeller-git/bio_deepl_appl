import torch
from torch import nn
from sklearn.model_selection import KFold
from tqdm import tqdm
from torch.utils.data import DataLoader, sampler
import numpy as np

# TODO fix circular import
from src import data_analysis
from src.modelling.models import (
    BasicMLP,
    MLP,
    Siamese,
    ExtendedSiamese,
    ModelParams,
    TriameseNetwork,
)
from src.modelling.eval import LossPlotter
from src.utils import (
    load_df,
    Plotter,
    save_model,
    ProtEmbeddingDataset,
    EmptyPlotter,
    weight_reset,
    save_params,
    cross_validate,
)

import src.utils as utils

global DEVICE
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_default_dtype(torch.float32)
torch.manual_seed(42)

print("Device set to", DEVICE)


class TrainParams:
    epochs: int
    out: str
    train_df: str
    lr: float
    batch_size: int
    cv: int
    kfold_args: dict
    # cv_def: CVDefinition

    def __init__(
        self,
        train_df: str = "./data/",
        out: str = "./out/",
        epochs: int = 10,
        lr: float = 1e-4,
        batch_size: int = 1024,
        cv: int = 0,
        # cv_definition: CVDefinition = CVDefinition(),
        **kwargs,
    ):
        self.epochs = epochs
        self.lr = lr
        self.train_df = train_df
        self.batch_size = batch_size
        self.cv = cv
        self.kfold_args = kwargs
        self.out = out

    def __repr__(self):
        return f"Params:\nepochs={self.epochs} | lr={self.lr} | batch_size={self.batch_size}"


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
    print("training model")
    train_loop(model, train_df, val_df, params, plotter)
    plotter.should_save("test")
    plotter.plot()
    save_model(model, params.out + "best_model.pth")
    save_params(params)
    print("model trained and saved")
    data_analysis.baseline(
        [model.cpu()], ["rmse", "spearman", "pearson"], val_df, p=params.train_df
    )
    utils.validate(model, val_df)

    cross_validate(model, val_df, params.train_df + "project_data/mega_val.csv")

    # VERY slow
    # cross_validate(model, train_df, params.train_df + "project_data/mega_train.csv")


def train_loop(
    model: nn.Module,
    train: DataLoader,
    test: DataLoader,
    params: TrainParams,
    plotter: Plotter = EmptyPlotter(),
):
    optim = torch.optim.Adam(model.parameters(), params.lr)
    criterion = RMSELoss(1e-6)
    scheduler_plat = torch.optim.lr_scheduler.ReduceLROnPlateau(optim, patience=2)
    scheduler2 = torch.optim.lr_scheduler.ExponentialLR(optim, 0.9)
    model.to(DEVICE)
    for epoch in tqdm(range(params.epochs), disable=VERBOSITY < 2, desc="train loop"):
        # print(f"Epoch {epoch}, Batches: {len(train)}")
        model.train()
        step(model, optim, criterion, train, plotter)
        model.eval()
        with torch.no_grad():
            print(f"\nEpoch {epoch}:")
            val_loss = validate(model, criterion, test, plotter)
        scheduler_plat.step(val_loss)
        scheduler2.step()
        print("\tlr:  ", scheduler2._last_lr)


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
    models = [
        TriameseNetwork(
            ModelParams(out_shape=256, n_layers=1, hidden_dim=256),
            ModelParams(n_layers=1, hidden_dim=256, out_shape=256),
            ModelParams(hidden_dim=512),
        ),
        TriameseNetwork(
            ModelParams(out_shape=256, n_layers=1, hidden_dim=512),
            ModelParams(n_layers=2, hidden_dim=512, out_shape=512),
            ModelParams(hidden_dim=512),
        ),
        TriameseNetwork(
            ModelParams(n_layers=2, hidden_dim=512, out_shape=512),
            ModelParams(out_shape=256, n_layers=1, hidden_dim=512),
            ModelParams(hidden_dim=512),
        ),
        TriameseNetwork(
            ModelParams(n_layers=2, out_shape=256),
            ModelParams(n_layers=2, out_shape=256),
            ModelParams(hidden_dim=512),
        ),
        TriameseNetwork(
            ModelParams(n_layers=2, out_shape=256, act=nn.ReLU),
            ModelParams(n_layers=2, out_shape=256, act=nn.ReLU),
            ModelParams(hidden_dim=512),
        ),
    ]
    val_df = np.zeros((len(models), params.cv))
    train_df = np.zeros((len(models), params.cv))
    plotter = LossPlotter()
    kfold_plotter = LossPlotter("rmse kfold", out=params.out)
    kfold_params = []
    for i in tqdm(range(len(models)), disable=VERBOSITY < 2, desc="cv"):
        base_lr = params.lr
        base_epochs = params.epochs
        base_batch_size = params.batch_size
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

    for split, (train_idc, val_idc) in enumerate(kf.split(train_data.ids)):
        print(f"split {split}")

        for m, model in enumerate(models):
            train_sampler = sampler.SubsetRandomSampler(train_idc)
            val_sampler = sampler.SubsetRandomSampler(val_idc)
            train_split = DataLoader(
                train_data,
                batch_size=kfold_params[m].batch_size,
                shuffle=False,
                num_workers=16,
                sampler=train_sampler,
            )
            val_split = DataLoader(
                train_data,
                batch_size=kfold_params[m].batch_size,
                shuffle=False,
                num_workers=16,
                sampler=val_sampler,
            )

            print(f"model {m}")
            train_loop(model, train_split, val_split, kfold_params[m], plotter)
            val_df[m, split] = sum(plotter.y["val loss"]) / len(plotter.y["val loss"])
            train_df[m, split] = sum(plotter.y["train loss"]) / len(
                plotter.y["train loss"]
            )
            plotter.clear()
            kfold_plotter.update("val model " + str(m), val_df[m, split], split)
            kfold_plotter.update("train model " + str(m), train_df[m, split], split)
            weight_reset(model)

    kfold_plotter.should_save("cv")
    kfold_plotter.plot()
    (train_df, train_std) = (np.mean(train_df, axis=1), np.std(train_df, axis=1))
    (val_df, val_std) = (np.mean(val_df, axis=1), np.std(val_df, axis=1))
    best_model = np.argmin(val_df)
    print(f"Val df: {val_df}")
    print(f"Train df: {train_df}")
    print(
        f"Best model: model {best_model}: {models[best_model]}\ntrain loss: {train_df[best_model]} | train std: {train_std[best_model]} | val loss: {val_df[best_model]} | val std: {val_std[best_model]}\nParams: {kfold_params[best_model]}"
    )
    train(models[best_model], kfold_params[best_model])


def validate(
    model: nn.Module, criterion: nn.Module, df: DataLoader, plotter: Plotter
) -> torch.Tensor:
    losses = []
    preds = []
    all_y = []
    for (embs, mut_embs), lbl in df:
        (embs, mut_embs, lbl) = (embs.to(DEVICE), mut_embs.to(DEVICE), lbl.to(DEVICE))
        yhat = model(embs, mut_embs).squeeze()
        loss = criterion(yhat, lbl)
        losses.append(loss)
        preds.append(yhat.cpu().detach().numpy())
        all_y.append(lbl.cpu().detach().numpy())

    all_y = np.concatenate(all_y)
    preds = np.concatenate(preds)
    loss = sum(losses) / len(losses)
    r = data_analysis.validate(all_y, preds, ["pearson", "spearman"], False)
    scc = r["Spearman Correlation"]
    pcc = r["Pearson Correlation"]

    plotter.update("val loss", loss.detach().cpu().numpy())
    print(f"Loss: {loss:.3f} | scc: {scc:.3f} | pcc: {pcc:.3f}")
    return loss


def step(
    model: nn.Module,
    optim: torch.optim.Optimizer,
    criterion: nn.Module(),
    df: DataLoader,
    plotter: Plotter,
):
    losses = []
    for (embs, mut_embs), lbl in df:
        (embs, mut_embs, lbl) = (embs.to(DEVICE), mut_embs.to(DEVICE), lbl.to(DEVICE))
        optim.zero_grad()
        yhat = model(embs, mut_embs).squeeze()
        loss = criterion(yhat, lbl)
        loss.backward()
        optim.step()
        losses.append(loss.detach())
    losses = sum(losses) / len(losses)
    print(f"\tTrain Loss: {losses:.3f}", end="\t")
    plotter.update("train loss", losses.cpu().numpy())
