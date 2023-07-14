import sys
import os 
sys.path.insert(1, os.path.join(sys.path[0], '../../src'))

import argparse
import logging
import json
from datetime import datetime

from sklearn.preprocessing import MinMaxScaler
from darts.dataprocessing.transformers.scaler import Scaler

from train_funcs import train_predict_arima, train_predict_baseline, train_predict_nhits_global, train_predict_transformer, train_predict_nhits_local
from features.encoders import past_datetime_encoder
from evaluation import evaluate
from datasets import ShellDataset, PaloAltoDataset, BoulderDataset, WeatherEcadDataset, load_dataset


logging.getLogger("pytorch_lightning").setLevel(logging.WARNING)


def main(args):
    series_dataset = load_dataset(args.dataset)
    series = series_dataset.load(
        subset=args.subset, train_length=args.train_data, test_length=args.test_data, na_threshold=0.1)
    
    # Scale series Data
    series_scaler = Scaler(MinMaxScaler())
    series_train = series_scaler.fit_transform(series['train'])
    series_test = series_scaler.transform(series['test'])

    # Load Weather Data
    if args.use_covariates:
        assert args.dataset == 'shell', 'Covariates are only available for Shell dataset.'
        weather_dataset = WeatherEcadDataset(value_cols=['temp_max']).from_series(series_train, series_test)

        # Scale series Data
        weather_scaler = Scaler(MinMaxScaler())
        weather_train = weather_scaler.fit_transform(weather_dataset['train'])
        weather_test = weather_scaler.transform(weather_dataset['test'])
    else:
        weather_train, weather_test = None, None

    # Use datetime features
    encoder = past_datetime_encoder if args.datetime_features else None

    # Train and Predict
    if args.model == 'Baseline':
        predictions = train_predict_baseline(
            series_train, series_test, args.forecast_horizon)
    elif args.model == 'ARIMA':
        predictions = train_predict_arima(
            series_train, series_test, args.forecast_horizon)
    elif args.model == 'Transformer':
        predictions = train_predict_transformer(
            series_train, series_test, encoder, args.input_chunk_length, args.forecast_horizon, weather_train, weather_test)
    elif args.model == 'NHiTS-Local':
        predictions = train_predict_nhits_local(
            series_train, series_test, encoder, args.input_chunk_length, args.forecast_horizon, weather_train, weather_test)
    elif args.model == 'NHiTS-Global':
        predictions = train_predict_nhits_global(
            series_train, series_test, encoder, args.input_chunk_length, args.forecast_horizon, weather_train, weather_test)
    else:
        raise "Model Not Found."

    # Scale back to original scale
    predictions = series_scaler.inverse_transform(predictions)

    # Evaluate
    scores = evaluate(predictions, series['test'])
    return scores


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Experimental Python script.")
    parser.add_argument("--dataset", type=str, default='paloalto',
                        help="Dataset: shell, paloalto, boulder")
    parser.add_argument("--model", type=str, default='Baseline', help="Available Models: Baseline, Transformer, ARIMA, NHiTS-Local, NHiTS-Global")
    parser.add_argument("--forecast_horizon", type=int,
                        default=1, help="Forecast horizon value")
    parser.add_argument("--input_chunk_length", type=int,
                        default=30, help="Input chunk length value")
    parser.add_argument("--use_covariates",
                        default=False,  action='store_true', help="Use covariates value")
    parser.add_argument("--datetime_features",
                        default=False, action='store_true', help="Use datetime features")
    parser.add_argument("--train_data", type=int,
                        default=240, help="Train data value")
    parser.add_argument("--test_data", type=int,
                        default=90, help="Test data value")
    parser.add_argument("--na_threshold", type=float,
                        default=0.1, help="Maximum allowed NA values in series")
    parser.add_argument("--subset", type=int,
                        default=None, help="Train data value")
    parser.add_argument("--seed", type=int, default=42, help="Random Seed")

    args = parser.parse_args()

    output = {'args': vars(args), 'results': {}}
    
    for model in ["Baseline", "ARIMA", "Transformer", "NHiTS-Local", "NHiTS-Global"]:
        args.model = model
        output["results"][model] = {}
        for forecast_horizon in [1, 7, 30]:
            args.forecast_horizon = forecast_horizon
            scores = main(args)
            output["results"][model][forecast_horizon] = scores

    time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    with open(f'results/{args.dataset}_results.json', 'w') as f:
        json.dump(output, f, indent=4)
