import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

from src import data_analysis
from src.utils import blosum


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
