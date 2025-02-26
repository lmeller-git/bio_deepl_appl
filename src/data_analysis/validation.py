import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import mean_squared_error


def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))


def pearson_corr(y_true, y_pred):
    corr, _ = pearsonr(y_true, y_pred)
    return corr


def spearman_corr(y_true, y_pred):
    corr, _ = spearmanr(y_true, y_pred)
    return corr


def validate(
    y_true, y_pred, performance_metric: list[str] = [], visualize: bool = True
):
    # return {"Pearson Correlation": 0, "Spearman Correlation": 0, "RMSE": 0}
    # TODO
    results = {}

    y_true = y_true.cpu().detach().numpy()
    y_pred = y_pred.cpu().detach().numpy()

    if performance_metric is None:
        performance_metric = ["rmse", "pearson", "spearman"]

    if "rmse" in performance_metric:
        results["RMSE"] = rmse(y_true, y_pred)

    if "pearson" in performance_metric:
        results["Pearson Correlation"] = pearson_corr(y_true, y_pred)

    if "spearman" in performance_metric:
        results["Spearman Correlation"] = spearman_corr(y_true, y_pred)

    if visualize:
        plot_predictions(y_true, y_pred)

    return results


def plot_predictions(y_true, y_pred, title="Predictions vs. True Values"):
    plt.figure(figsize=(14, 6))

    plt.subplot(1, 2, 1)
    sns.regplot(
        x=y_true, y=y_pred, scatter_kws={"alpha": 0.6}, line_kws={"color": "red"}
    )
    plt.xlabel("True Values")
    plt.ylabel("Predicted Values")
    plt.title(f"{title} (Scatter Plot)")

    residuals = y_true - y_pred
    plt.subplot(1, 2, 2)
    sns.histplot(residuals, kde=True, color="skyblue")
    plt.axvline(0, color="red", linestyle="--")
    plt.xlabel("Residuals")
    plt.title(f"{title} (Residual Distribution)")

    plt.tight_layout()
    plt.show()
