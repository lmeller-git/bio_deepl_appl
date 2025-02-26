import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

from src import data_analysis
from src.utils import blosum

def compare_to_baseline(
        y_true, model_preds: dict = {}, performance_metric: list[str] = [], visualize: bool = True
):

    baseline_pred = blosum.main()[0]  

    results = {"Baseline": data_analysis.validate(y_true, baseline_pred, performance_metric=performance_metric, visualize=False)}

    comparison_results = {}

    for name, y_pred in model_preds.items():
        print(f"\nValidating Model: {name}")
        model_results = data_analysis.validate(y_true, y_pred, performance_metric=performance_metric, visualize=False)
        
        diff_results = {performance_metric: model_results[performance_metric] - baseline_results[performance_metric] for performance_metric in model_results}
        comparison_results[name] = diff_results
        
        print(f"Results for {name}: {model_results}")
        print(f"Difference from Baseline: {diff_results}")
        
        results[name] = model_results
        

    print("\nPerformance Metrics:")
    for model_name, metrics_dict in results.items():
        print(f"\n{model_name}:")
        for metric, value in metrics_dict.items():
            print(f"  {metric}: {value:.4f}")

    if visualize:
        results_df = pd.DataFrame(results).T
        plt.figure(figsize=(10, 7))
        sns.heatmap(results_df, annot=True, cmap="viridis", fmt=".3f", cbar=True)
        plt.title("Model Performance Metrics")
        plt.show()

    return results
