#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 25 16:22:08 2025

@author: takutaku
"""


    

def compare_models(models, x_test, y_test, metrics=None, visualize=True):
    """
    Compare multiple models based on validation metrics and visualization.

    Parameters:
    - models: dict, model names as keys and trained models as values.
    - x_test: array-like, test input data.
    - y_test: array-like, true output values.
    - metrics: list of strings, metrics to calculate ('rmse', 'pearson', 'spearman').
    - visualize: bool, whether to visualize results.

    Returns:
    - Dictionary with model names as keys and validation results as values.
    """
    comparison_results = {}

    for name, model in models.items():
        print(f"\nValidating Model: {name}")
        y_pred = model.predict(x_test)
        results = validate(y_test, y_pred, metrics=metrics, visualize=visualize)
        comparison_results[name] = results

        print(f"Results for {name}: {results}")

    return comparison_results