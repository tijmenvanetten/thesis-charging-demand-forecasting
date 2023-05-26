from data import load_target, load_covariates, split_data
from models import load_model, evaluate
from visualization import plot_separate
from features.clustering import stack_timeseries

from darts import concatenate
from darts.dataprocessing.transformers.scaler import Scaler

from sklearn.preprocessing import MinMaxScaler
import argparse
import logging
import os
from typing import Dict
from datetime import datetime


def run(timeseries, covariates, args) -> Dict:
    """
    runs experiment

    """

    print(list(len(series) for series in timeseries))
    timeseries = stack_timeseries(timeseries)
    print(list(len(series) for series in timeseries))

    # # custom selection of timeseries
    # timeseries = [series for series in target_series if len(
    #     series) == 1035]
    # indices = [0, 1, 2, 4, 10, 11, 14, 15, 16, 17, 18, 20]
    # timeseries = [timeseries[i] for i in indices][:args.subset]
    

    if args.train_global:
        timeseries = [concatenate(timeseries, axis=1, ignore_time_axis=True)]

    # Traing and evaluate per timeseries
    predictions = []
    for series in timeseries:
        train, val, test = split_data(series, args.train_split, args.val_split)
        covariates_train, covariates_val, covariates_test = split_data(
            covariates, args.train_split, args.val_split)

        if args.use_covariates:
            # align covariates to target
            covariates = covariates[series.time_index]

        # scale data
        if args.scale:
            # scale training data
            target_scaler = Scaler(MinMaxScaler())
            train = target_scaler.fit_transform(train)
            val = target_scaler.transform(val)
            series = target_scaler.transform(series)

            covariates_scaler = Scaler(MinMaxScaler())
            covariates_train = covariates_scaler.fit_transform(
                covariates_train)
            covariates_val = covariates_scaler.transform(covariates_val)
            covariates = covariates_scaler.transform(covariates)

        # load model
        model = load_model(args)

        # train model on training data
        training_args, eval_args = {}, {}
        if args.use_covariates:
            training_args['past_covariates'] = covariates, 
            training_args['val_past_covariates'] = covariates
            eval_args['past_covariates'] = covariates

        if args.use_val:
            training_args['val_series'] = val

        # evaluate on all data
        eval_args = {}
        if args.retrain:
            eval_args['retrain'] = True
        else:
            eval_args['retrain'] = False

        model.fit(
            series=train,
            **training_args
        )
        
        backtest = model.historical_forecasts(
            series=series,
            num_samples=args.num_samples,
            start=test.start_time(),
            forecast_horizon=args.forecast_horizon,
            verbose=False,
            **eval_args,
        )

        prediction = target_scaler.inverse_transform(backtest)
        predictions.append(prediction)

    logging.info(f"Finished training")
    # Log Metrics
    scores = evaluate(predictions, timeseries)
    logging.info(f"results: {scores}")

    # Save Forecast plot
    figs = plot_separate(predictions, timeseries)
    for location_id, fig in figs.items():
        fig.savefig(args.logdir + "plots/" + f"{location_id.replace('/', '_')}.png")


if __name__ == "__main__":

    logging.getLogger("pytorch_lightning").setLevel(logging.WARNING)

    parser = argparse.ArgumentParser(description="Argument Parser")

    # Data options
    parser.add_argument("--train_global", action='store_true')
    parser.add_argument("--subset", type=int, default=None)
    parser.add_argument("--use_covariates", action='store_true')
    parser.add_argument("--use_val", action='store_true')
    parser.add_argument("--scale", action='store_false')

    # Training options
    parser.add_argument("--model", type=str, default="NBEATS",
                        help="a string specifying the model")
    parser.add_argument("--train_split", type=float, default=0.7)
    parser.add_argument("--val_split", type=float, default=0.8)

    # Inference options
    parser.add_argument("--retrain", action='store_true')
    parser.add_argument("--forecast_horizon", type=int, default=1)
    parser.add_argument("--num_samples", type=int, default=1)

    args = parser.parse_args()
    print(args)

    start_time_str = datetime.today().strftime('%d-%m-%Y_%H-%M-%S')
    args.logdir = f"logs/{start_time_str}_{args.model=}_{args.train_global=}_{args.use_covariates=}/"
    os.mkdir(args.logdir)
    os.mkdir(args.logdir + "plots/")

    logging.basicConfig(filename=args.logdir + "logs.log",
                        encoding='utf-8', level=logging.INFO)

    # # Load Data
    # target_series = load_target(
    #     path='data/03_processed/on_forecourt_sessions.csv',
    #     group_cols='location_id',
    #     time_col='date',
    #     value_cols='energy_delivered_kwh',
    #     static_cols=['num_evse'],
    #     freq='D'
    # )
    # Load Data
    target_series = load_target(
        path='data/03_processed/palo_alto_dataset.csv',
        group_cols='location_id',
        time_col='date',
        value_cols='energy_delivered_kwh',
        # static_cols=['num_evse'],
        freq='D'
    )

    covariates = load_covariates(
        path='data/03_processed/weather_ecad.csv',
        time_col='date',
        value_cols=['temp_max', 'temp_min', 'sunshine', 'precip'],
        freq='D'
    )

    logging.info(f"Running Experiments, params: {args}")
    run(target_series, covariates, args)
