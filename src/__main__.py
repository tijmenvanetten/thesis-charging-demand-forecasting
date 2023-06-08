from data import split_data, data_handler
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

def main(target_data, covariates_data, args, logger) -> Dict:
    """
    Runs an experiment using the provided target and covariates.

    Args:
        target_data (List[TimeSeries]): The list of target TimeSeries objects.
        covariates_data (TimeSeries): The covariates TimeSeries object.
        args: The arguments for the experiment.
        logger: The logger object for logging messages.

    Returns:
        Dict: A dictionary containing the predictions and figures generated during the experiment.

    Example:
        >>> target_data = [series1, series2, ...]
        >>> covariates_data = covariates_series
        >>> args = ...
        >>> logger = ...
        >>> results = main(target_data, covariates_data, args, logger)
    """
    target_series = [series for series in target_data if len(series) == 1035]
    target_series = target_series[:args.subset]

    if args.train_global:
        # Concatenate target series along the axis=1
        target_series = [concatenate(target_series, axis=1)]

    predictions = []
    for target_series_single in target_series:
        train, val, test = split_data(target_series_single, args.train_split, args.val_split)

        if args.use_covariates:
            covariates_train, covariates_val, covariates_test = split_data(
                covariates_data, args.train_split, args.val_split)
            covariates_data = covariates_data[target_series_single.time_index]

        if args.scale:
            target_scaler = Scaler(MinMaxScaler())
            train = target_scaler.fit_transform(train)
            val = target_scaler.transform(val)
            target_series_single = target_scaler.transform(target_series_single)

            if args.use_covariates:
                covariates_scaler = Scaler(MinMaxScaler())
                covariates_train = covariates_scaler.fit_transform(
                    covariates_train)
                covariates_val = covariates_scaler.transform(covariates_val)
                covariates_data = covariates_scaler.transform(covariates_data)

        print("Loading New model")
        model = load_model(args)

        train_args, eval_args = {}, {}
        if args.use_covariates:
            train_args['past_covariates'] = covariates_data,
            train_args['val_past_covariates'] = covariates_data
            eval_args['past_covariates'] = covariates_data
        if args.use_val:
            train_args['val_series'] = val
        if args.retrain:
            eval_args['retrain'] = True
        else:
            eval_args['retrain'] = False

        model.fit(
            series=train,
            **train_args
        )

        backtest = model.historical_forecasts(
            series=target_series_single,
            num_samples=args.num_samples,
            start=test.start_time(),
            forecast_horizon=args.forecast_horizon,
            verbose=False,
            **eval_args,
        )

        prediction = target_scaler.inverse_transform(backtest)
        predictions.append(prediction)

    evaluation_results = evaluate(predictions, target_series)

    forecast_plots = plot_separate(predictions, target_series)
    for idx, fig in forecast_plots.items():
        fig.savefig(args.logdir + "plots/" + f"{idx=}.png")

    return evaluation_results, predictions, forecast_plots


if __name__ == "__main__":

    logging.getLogger("pytorch_lightning").setLevel(logging.WARNING)

    parser = argparse.ArgumentParser(description="Argument Parser")

    # Data options
    parser.add_argument("--dataset", type=str, default='shell')
    parser.add_argument("--subset", type=int, default=None)
    parser.add_argument("--use_covariates", action='store_true')
    parser.add_argument("--use_val", action='store_true')
    parser.add_argument("--use_static_cols", action='store_true')

    # Training options
    parser.add_argument("--model", type=str,
                        help="a string specifying the model")
    parser.add_argument("--train_global", action='store_true')
    parser.add_argument("--scale", action='store_false')
    parser.add_argument("--train_split", type=float, default=0.7)
    parser.add_argument("--val_split", type=float, default=0.8)

    # Evaluation options
    parser.add_argument("--retrain", action='store_true')
    parser.add_argument("--forecast_horizon", type=int, default=1)
    parser.add_argument("--num_samples", type=int, default=1)

    args = parser.parse_args()

    start_time_str = datetime.today().strftime('%d-%m-%Y, %H:%M:%S')
    log_folder = f"{args.model}/Global={args.train_global}, Covariates={args.use_covariates}, Retrain={args.retrain}"
    args.logdir = f"logs/{log_folder}/{start_time_str}/"
    os.makedirs(args.logdir, exist_ok=True)
    os.makedirs(args.logdir + "plots/", exist_ok=True)

    # Main logger for general logging
    logging.basicConfig(filename=args.logdir + "logs.log",
                        encoding='utf-8', level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    # Separate logger for parameters and results only
    results_logger = logging.getLogger("results_logger")
    results_logger.setLevel(logging.INFO)
    results_handler = logging.FileHandler(args.logdir + "results.log")
    results_logger.addHandler(results_handler)
    
    # Load Data
    dataset = data_handler(args.dataset, args.use_static_cols)
    target_series = dataset.get('target')
    covariates = dataset.get('covariates')

    # Run Experiments
    logger.info(f"{start_time_str}: Running Experiments, params: {args}")
    evaluation_results, predictions, forecast_plots = main(target_series, covariates, args, logger)

    # Log Results
    results_string = json.dumps(evaluation_results, indent=4)
    results_logger.info(f"Results: {results_string}")
