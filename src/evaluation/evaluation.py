from typing import Dict, List
from darts.metrics import mape, rmse, mse, mae
from darts import TimeSeries
import numpy as np

def evaluate(predictions: List[TimeSeries], actuals: List[TimeSeries]) -> Dict[str, float]:
    """
    Evaluate the performance of predictions against actuals using weighted metrics.

    Args:
        predictions (List[TimeSeries]): List of predicted TimeSeries objects.
        actuals (List[TimeSeries]): List of actual TimeSeries objects.

    Returns:
        Dict[str, float]: A dictionary containing weighted MAPE, RMSE, MSE, and MAE metrics.
    """
    # Calculate individual metrics for each prediction-actual pair
    mape_values = mape(predictions, actuals)
    rmse_values = rmse(predictions, actuals)
    mse_values = mse(predictions, actuals)
    mae_values = mae(predictions, actuals)

    # Calculate weights based on the length of actuals
    total_length = sum(len(actual) for actual in actuals)
    weights = [len(actual) / total_length for actual in actuals]

    # Calculate weighted mean of metrics using the weights
    mape_value = np.average(mape_values, weights=weights)
    rmse_value = np.average(rmse_values, weights=weights)
    mse_value = np.average(mse_values, weights=weights)
    mae_value = np.average(mae_values, weights=weights)

    # Create a dictionary with the weighted metrics
    weighted_metrics = {
        'RMSE': rmse_value,
        'MAE': mae_value,
        'MAPE': mape_value,
    }

    return weighted_metrics


def print_metrics_table(predictions_dict: Dict[str, List[TimeSeries]], actuals: List[TimeSeries]):
    """
    Prints the metrics for each model in a nice aligned table format.

    Args:
        predictions_dict (Dict[str, List[TimeSeries]]): Dictionary with model names as keys and predictions as values.
        actuals (List[TimeSeries]): List of actual TimeSeries objects.
    """
    # Calculate metrics for each model's predictions
    metrics = []
    metrics_keys = ['RMSE', 'MAE', 'MAPE']
    for model, predictions in predictions_dict.items():
        weighted_metrics = evaluate(predictions, actuals)
        metric_row = [model] + [weighted_metrics[key] for key in metrics_keys]
        metrics.append(metric_row)

    # Define table format
    table_format = "{:<15}" + "{:<10}" * (len(metrics_keys) + 1)

    # Print table headers
    header = ["Model"] + metrics_keys
    print(table_format.format(*header))
    print("-" * (15 + 10 * (len(metrics_keys) + 1)))

    # Print metrics table
    for metric_row in metrics:
        print(table_format.format(*metric_row))