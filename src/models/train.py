from data import train_val_test_split, data_handler
from models import load_model
from eval import evaluate
from visualization import plot_separate

from darts import concatenate
from darts.dataprocessing.transformers.scaler import Scaler
from sklearn.preprocessing import MinMaxScaler

import argparse
import logging
import os
import json
from typing import Dict
from datetime import datetime

from typing import Dict


def train_predict(train_series, test_series, split_val, load_model_func, train_global, forecast_horizon, scale, retrain, num_samples, logdir=None) -> Dict:
    """
    Runs an experiment using the provided target and covariates.

    Args:
        target_data (List[TimeSeries]): The list of target TimeSeries objects.
        covariates_data (TimeSeries): The covariates TimeSeries object.
        args: The arguments for the experiment.

    Returns:
        Dict: A dictionary containing the predictions and figures generated during the experiment.

    Example:
        >>> target_data = [series1, series2, ...]
        >>> covariates_data = covariates_series
        >>> results = main(target_data, covariates_data, args)
    """

    series = concatenate([train_series, test_series])
    if split_val:
        train_series, val_series = train_series.split_after(split_val)
    else:
        train_series, val_series = train_series, [None] * len(train_series)
        
    if scale:
        target_scaler = Scaler(MinMaxScaler(), global_fit=train_global)
        train_series = target_scaler.fit_transform(train_series)
        val_series = target_scaler.transform(
            val_series) if split_val else val_series
        
    if not train_global:
        # Split Time series into multiple univariate components
        train_series = [train_series.univariate_component(i) for i in range(train_series.n_components)]
        val_series = [val_series.univariate_component(i) for i in range(val_series.n_components)] if split_val else val_series
    else:
        train_series = [train_series]
        val_series = [val_series]

    if isinstance(forecast_horizon, int):
        forecast_horizon = [forecast_horizon]  # Convert single integer to a list

    predictions = {}
    for i, (train_series_single, valid_series_single) in enumerate(zip(train_series, val_series)):
        print("Loading New model")
        model = load_model_func()

        train_args, eval_args = {}, {}
        if valid_series_single:
            train_args['val_series'] = valid_series_single
        if retrain:
            eval_args['retrain'] = True
        else:
            eval_args['retrain'] = False

        model.fit(
            series=train_series_single,
            **train_args
        )

        forecast_dict = {}
        for horizon in forecast_horizon:
            forecast = model.historical_forecasts(
                series=series.univariate_component(i),
                num_samples=num_samples,
                start=test_series.start_time(),
                forecast_horizon=horizon,
                verbose=False,
                overlap_end=False,
                **eval_args,
            )
            forecast_dict[horizon] = forecast

        predictions[i] = forecast_dict

    if scale:
        # Inverse transform the predictions
        for i, forecast_dict in predictions.items():
            for horizon, forecast in forecast_dict.items():
                forecast = target_scaler.inverse_transform(forecast)
                forecast_dict[horizon] = forecast

    return predictions

