from darts import concatenate
from darts.dataprocessing.transformers.scaler import Scaler
from sklearn.preprocessing import MinMaxScaler

from typing import Dict


def train_predict(train_series, test_series, split_val, load_model_func, train_global, forecast_horizon, scale, num_samples, past_covariates=None) -> Dict:
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

    if split_val:
        train_series, val_series = train_series.split_after(split_val)
    else:
        train_series, val_series = train_series, [None] * len(train_series)
        
    if scale:
        target_scaler = Scaler(MinMaxScaler(), global_fit=train_global)
        train_series = target_scaler.fit_transform(train_series)
        val_series = target_scaler.transform(
            val_series) if split_val else val_series
        
        if past_covariates:
            covariates_scaler = Scaler(MinMaxScaler())
            past_covariates = covariates_scaler.fit_transform(past_covariates)

    if not train_global:
        # Split Time series into multiple univariate components
        train_series = [train_series.univariate_component(i) for i in range(train_series.n_components)]
        val_series = [val_series.univariate_component(i) for i in range(val_series.n_components)] if split_val else val_series
        test_series = [test_series.univariate_component(i) for i in range(test_series.n_components)]
    else:
        train_series = [train_series]
        val_series = [val_series]
        test_series = [test_series]

    if isinstance(forecast_horizon, int):
        forecast_horizon = [forecast_horizon]  # Convert single integer to a list

    predictions = {horizon: [] for horizon in forecast_horizon}
    for i, (train_series_single, valid_series_single, test_series_single) in enumerate(zip(train_series, val_series, test_series)):
        print("Loading New model")
        model = load_model_func()

        train_args, eval_args = {}, {}
        if valid_series_single:
            train_args['val_series'] = valid_series_single
            if past_covariates:
                train_args['val_past_covariates'] = past_covariates

        model.fit(
            series=train_series_single,
            past_covariates=past_covariates if past_covariates else None,
            **train_args
        )

        if valid_series_single:
            full_series = concatenate([valid_series_single, test_series_single], axis=0) 
        else:
            full_series = concatenate([train_series_single, test_series_single], axis=0)

        for horizon in forecast_horizon:
            forecast = model.historical_forecasts(
                series=full_series,
                num_samples=num_samples,
                start=test_series_single.start_time(),
                forecast_horizon=horizon,
                past_covariates=past_covariates if past_covariates else None,
                verbose=False,
                retrain=False,
                overlap_end=False,
                **eval_args,
            )
            predictions[horizon].append(forecast)

    for horizon, forecasts in predictions.items():
        forecasts = concatenate(forecasts, axis=1)
        if scale:
            forecasts = target_scaler.inverse_transform(forecasts)
        predictions[horizon] = forecasts
    return predictions

