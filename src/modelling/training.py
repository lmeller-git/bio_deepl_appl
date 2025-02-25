import torch
from torch import nn
from sklearn.model_selection import KFold
from tqdm import tqdm
from src.given_utils import load_df
from models import BasicMLP

global DEVICE
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.set_default_dtype(torch.float32)
torch.manual_seed(42)


class TrainParams:
    epochs: int
    train_df: str
    lr: float
    cv: int
    kfold_args: dict

    def __init__(
        self, train_df, epochs: int = 10, lr: float = 1e-4, cv: int = 0, **kwargs
    ):
        self.epochs = epochs
        self.lr = lr
        self.train_df = train_df
        self.cv = cv
        self.kfold_args = kwargs


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

    # TODO implement kfold integration

    train_df, val_df, test_df = load_df(params.train_df)

    # model = BasicMLP(768)
    optim = torch.optim.Adam(model.parameters(), params.lr)
    criterion = RMSELoss(0)

    for epoch in tqdm(range(params.epochs)):
        model.train()
        step(model, optim, criterion, train_df)
        model.eval()
        with torch.no_grad():
            print(f"Epoch {epoch}:")
            validate(model, criterion, val_df)


def kfold(params: TrainParams) -> nn.Module:
    pass


def validate(model: nn.Module, criterion: nn.Module, df):
    losses = []
    scc = []
    pcc = []
    for embs, lbl in df:
        (embs, lbl) = (embs.to(DEVICE), lbl.to(DEVICE))
        yhat = model(embs)
        loss = criterion(yhat, lbl)
        losses.append(loss)

    loss = sum(losses) / len(losses)

    print(f"\tLoss: {loss:.3f} | scc: {scc:.3f} | pcc: {pcc:.3f}")


def step(model: nn.Module, optim: torch.optim.Optimizer, criterion: nn.Module(), df):
    for embs, lbl in df:
        (embs, lbl) = (embs.to(DEVICE), lbl.to(DEVICE))
        optim.zero_grad()
        criterion.zero_grad()
        yhat = model(embs)
        loss = criterion(yhat, lbl)
        loss.backward()
        optim.step()
