import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from torch import nn
import torch
import torch.utils.data as tdata
from src import data_analysis
from src.utils import blosum, Plotter
from collections import defaultdict


class ComparisonResult:
    rmse: None | float
    scc: None | float
    pcc: None | float

    def __init__(self, rmse: float = None, scc: float = None, pcc: float = None):
        self.pcc = pcc
        self.scc = scc
        self.rmse = rmse

    def __add__(self, rhs):
        n = ComparisonResult()
        n.rmse = (
            self.rmse + rhs.rmse
            if self.rmse is not None
            else rhs.rmse
            if rhs.rmse is not None
            else None
        )
        n.scc = (
            self.scc + rhs.scc
            if self.scc is not None
            else rhs.scc
            if rhs.scc is not None
            else None
        )
        n.pcc = (
            self.pcc + rhs.pcc
            if self.pcc is not None
            else rhs.pcc
            if rhs.pcc is not None
            else None
        )
        return n

    def __sub__(self, rhs):
        n = ComparisonResult()
        n.rmse = (
            self.rmse - rhs.rmse
            if self.rmse is not None
            else rhs.rmse
            if rhs.rmse is not None
            else None
        )
        n.scc = (
            self.scc - rhs.scc
            if self.scc is not None
            else rhs.scc
            if rhs.scc is not None
            else None
        )
        n.pcc = (
            self.pcc - rhs.pcc
            if self.pcc is not None
            else rhs.pcc
            if rhs.pcc is not None
            else None
        )
        return n

    def div(self, rhs):
        self.pcc = self.pcc / rhs if self.pcc is not None else None
        self.scc = self.scc / rhs if self.scc is not None else None
        self.rmse = self.rmse / rhs if self.rmse is not None else None
        return self

    def __repr__(self):
        s = ""
        if self.pcc is not None:
            s += f"pcc: {self.pcc:.3f}\t"
        if self.scc is not None:
            s += f"scc: {self.scc:.3f}\t"
        if self.rmse is not None:
            s += f"rmse: {self.rmse:.3f}\t"
        return s


class ComparisonPlotter(Plotter):
    y: dict[ComparisonResult | list[ComparisonResult]]

    def __init__(self):
        super().__init__()
        self.y = defaultdict(list)

    def update(self, key: str, r: ComparisonResult):
        self.y[key].append(r)

    def clear(self):
        self.y = defaultdict(ComparisonResult)

    def finalize(self):
        y = defaultdict(ComparisonResult)
        for k, v in self.y.items():
            for v_ in v:
                y[k] += v_

            y[k] = y[k].div(len(v))
        self.y = y

    def plot(self):
        if not isinstance(self.y, dict):
            print("Expected a dictionary")
            return self.y

        first_value = next(iter(self.y.values()), None)
        if isinstance(first_value, list):
            print("Expected dict[ComparisonResult] after finalization")
            return self.y

        if not all(isinstance(v, ComparisonResult) for v in self.y.values()):
            print("Expected dict[ComparisonResult] after finalization")
            return self.y

        metrics = ["rmse", "scc", "pcc"]
        models = list(self.y.keys())
        values = np.array(
            [[self.y[m].rmse, self.y[m].scc, self.y[m].pcc] for m in models]
        )

        x = np.arange(len(metrics))
        width = 0.2

        fig, ax = plt.subplots(figsize=(8, 5))

        for i, model in enumerate(models):
            ax.bar(x + i * width, values[i], width, label=model)

        ax.set_xticks(x + width * (len(models) - 1) / 2)
        ax.set_xticklabels(metrics)
        ax.legend()
        ax.set_ylabel("Metric Value")
        ax.set_title("Model Performance Comparison")
        plt.show()

    def __repr__(self):
        s = ""
        baseline = self.y["baseline"]
        for k, v in self.y.items():
            s += f"{k}: {v}|\tdelta: {v - baseline}\n"

        return s


def baseline(
    models: list[nn.Module],
    callbacks: list[str],
    test_data: tdata.DataLoader,
    plotter: ComparisonPlotter = ComparisonPlotter(),
    p: str = "./data/",
):
    (baseline_pred, baseline_truth) = blosum.main(p)
    for r, t in zip(baseline_pred, baseline_truth):
        val_data = data_analysis.validate(
            torch.tensor(t), torch.tensor(r), callbacks, visualize=False
        )
        plotter.update(
            "baseline",
            ComparisonResult(
                rmse=val_data["RMSE"] if "RMSE" in val_data.keys() else None,
                scc=val_data["Spearman Correlation"]
                if "Spearman Correlation" in val_data.keys()
                else None,
                pcc=val_data["Pearson Correlation"]
                if "Pearson Correlation" in val_data.keys()
                else None,
            ),
        )
    for (embs, embs_mut), lbl in test_data:
        embs.cpu()
        lbl.cpu()
        embs_mut.cpu()
        for i, model in enumerate(models):
            yhat = model(embs, embs_mut).squeeze()
            val_data = data_analysis.validate(
                lbl, yhat, performance_metric=callbacks, visualize=False
            )
        plotter.update(
            "model " + str(i),
            ComparisonResult(
                rmse=val_data["RMSE"] if "RMSE" in val_data.keys() else None,
                scc=val_data["Spearman Correlation"]
                if "Spearman Correlation" in val_data.keys()
                else None,
                pcc=val_data["Pearson Correlation"]
                if "Pearson Correlation" in val_data.keys()
                else None,
            ),
        )

    plotter.finalize()
    print(plotter)
    plotter.plot()


def compare_to_baseline(
    y_true,
    model_preds: dict = {},
    performance_metric: list[str] = [],
    visualize: bool = True,
    p: str = "./data/",
):
    (baseline_pred, baseline_truth) = blosum.main(p)

    results = {
        "Baseline": data_analysis.validate(
            baseline_truth,
            baseline_pred,
            performance_metric=performance_metric,
            visualize=False,
        )
    }

    comparison_results = {}

    for name, y_pred in model_preds.items():
        print(f"\nValidating Model: {name}")
        model_results = data_analysis.validate(
            y_true, y_pred, performance_metric=performance_metric, visualize=False
        )

        diff_results = {m: model_results[m] - r for m, r in model_results.items()}
        comparison_results[name] = diff_results

        results[name] = model_results

    print("\nPerformance Metrics:")
    for (model_name, metrics_dict), (_, diff_dict) in zip(
        results.items(), comparison_results.iter()
    ):
        print(f"\n{model_name}:")
        for metric, value in metrics_dict.items():
            print(f"\t{metric}: {value:.3f}")
            print(f"\t{metric} delta: {diff_dict[metric]:.3f}")

    if visualize:
        results_df = pd.DataFrame(results).T
        plt.figure(figsize=(10, 7))
        sns.heatmap(results_df, annot=True, cmap="viridis", fmt=".3f", cbar=True)
        plt.title("Model Performance Metrics")
        plt.show()

    return results
