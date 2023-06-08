from typing import List 
from darts.metrics import mse, rmse, mae, mape
import matplotlib.pyplot as plt
import numpy as np
from darts import TimeSeries


import numpy as np

def evaluate(predictions, series_clusters):
    """
    Evaluates the predictions against the actual values in the series clusters and returns the evaluation results.

    Args:
        predictions (List[Forecast]): A list of Forecast objects containing the predicted values.
        series_clusters (List[TimeSeries]): A list of TimeSeries objects representing the series clusters.

    Returns:
        dict: A dictionary containing the evaluation results, including mean and standard deviation for each metric.

    Example:
        >>> preds = [forecast1, forecast2, ...]
        >>> clusters = [series_cluster1, series_cluster2, ...]
        >>> evaluation_results = evaluate(preds, clusters)
    """
    results = {}
    mse_total, rmse_total, mae_total, mape_total = [], [], [], []

    for forecast_cluster, series_cluster in zip(predictions, series_clusters):
        components = forecast_cluster.components

        for component in components:
            forecast = forecast_cluster.univariate_component(component)
            actual = series_cluster.univariate_component(component)[forecast_cluster.time_index]
            mse_total.append(mse(forecast, actual))
            rmse_total.append(rmse(forecast, actual))
            mae_total.append(mae(forecast, actual))
            mape_total.append(mape(forecast, actual))

    mse_total = np.array(mse_total)
    rmse_total = np.array(rmse_total)
    mae_total = np.array(mae_total)
    mape_total = np.array(mape_total)

    results['mape'] = {'mean': np.mean(mape_total), 'std': np.std(mape_total)}
    results['mse'] = {'mean': np.mean(mse_total), 'std': np.std(mse_total)}
    results['rmse'] = {'mean': np.mean(rmse_total), 'std': np.std(rmse_total)}
    results['mae'] = {'mean': np.mean(mae_total), 'std': np.std(mae_total)}

    return results
