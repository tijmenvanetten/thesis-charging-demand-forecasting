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
from typing import Dict
from datetime import datetime


def main(timeseries, covariates, args, logger) -> Dict:
    """
    runs experiment

    """
    # # custom selection of timeseries
    timeseries = [series for series in target_series if len(
        series) == 1035]
    # indices = [0, 1, 2, 4, 10, 11, 14, 15, 16, 17, 18, 20]
    # timeseries = [timeseries[i] for i in indices]

    timeseries = timeseries[:args.subset]
    if args.train_global:
        # timeseries = [stack_timeseries(timeseries)]
        timeseries = [concatenate(timeseries, axis=1)]
        # timeseries = [timeseries.univariate_component(component) for component in timeseries.components]
        
    # Traing and evaluate per timeseries
    predictions = []
    for series in timeseries:
        train, val, test = split_data(series, args.train_split, args.val_split)
        if args.use_covariates:
            covariates_train, covariates_val, covariates_test = split_data(
            covariates, args.train_split, args.val_split)
            # align covariates to target
            covariates = covariates[series.time_index]

        # scale data
        if args.scale:
            # scale training data
            target_scaler = Scaler(MinMaxScaler())
            train = target_scaler.fit_transform(train)
            val = target_scaler.transform(val)
            series = target_scaler.transform(series)

            if args.use_covariates:
                covariates_scaler = Scaler(MinMaxScaler())
                covariates_train = covariates_scaler.fit_transform(
                    covariates_train)
                covariates_val = covariates_scaler.transform(covariates_val)
                covariates = covariates_scaler.transform(covariates)

        # load model
        print("Loading New model")
        model = load_model(args)

        # train model on training data
        train_args, eval_args = {}, {}
        if args.use_covariates:
            train_args['past_covariates'] = covariates, 
            train_args['val_past_covariates'] = covariates
            eval_args['past_covariates'] = covariates
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
        
        # Set number of epochs to train as hyperparameter from training
        # model.n_epochs = model.epochs_trained

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

    logger.info(f"Finished training")
    # Log Metrics
    scores = evaluate(predictions, timeseries)
    logger.info(f"results: {scores}")

    # Save Forecast plot
    figs = plot_separate(predictions, timeseries)
    for idx, fig in figs.items():
        fig.savefig(args.logdir + "plots/" + f"{idx=}.png")

    return predictions, figs

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

    start_time_str = datetime.today().strftime('%d-%m-%Y, %H-%M')
    args.logdir = f"logs/{args.model=}_{args.train_global=}_{args.use_covariates=}_{args.retrain=}/"
    if not os.path.exists(args.logdir):
        os.mkdir(args.logdir)
        os.mkdir(args.logdir + "plots/")

    logging.basicConfig(filename=args.logdir + "logs.log",
                        encoding='utf-8', level=logging.INFO)
    logger = logging.getLogger(__name__)

    # Load Data
    dataset = data_handler(args.dataset, args.use_static_cols)
    target_series = dataset.get('target')
    covariates = dataset.get('covariates')

    logger.info(f"{start_time_str}: Running Experiments, params: {args}")
    main(target_series, covariates, args, logger)
