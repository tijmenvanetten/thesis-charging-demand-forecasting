from train import train_predict_arima, train_predict_baseline, train_predict_nhits_global, train_predict_transformer, train_predict_nhits_local
from evaluation import evaluate
from darts.dataprocessing.transformers.scaler import Scaler
from datasets import ShellDataset, PaloAltoDataset, BoulderDataset
import sys
import os
import json
import logging
import argparse
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from sklearn.preprocessing import MinMaxScaler

sys.path.insert(1, os.path.join(sys.path[0], '../../src'))
logging.getLogger("pytorch_lightning").setLevel(logging.WARNING)


def main(args):
    # Load Dataset
    if args.dataset == 'shell':
        series_dataset = ShellDataset()
    elif args.dataset == 'paloalto':
        series_dataset = PaloAltoDataset()
    elif args.dataset == 'boulder':
        series_dataset = BoulderDataset()
    else:
        raise Exception('Invalid dataset.')
    series = series_dataset.load(
        subset=args.subset, train_length=args.train_data, test_length=args.test_data, na_threshold=0.1)

    # Scale series Data
    series_scaler = Scaler(MinMaxScaler())
    series_train = series_scaler.fit_transform(series['train'])
    series_test = series_scaler.transform(series['test'])

    if args.model == 'Baseline':
        predictions = train_predict_baseline(series_train, series_test, args.forecast_horizon)
    elif args.model == 'ARIMA':
        predictions = train_predict_arima(series_train, series_test, args.forecast_horizon)
    elif args.model == 'Transformer':
        predictions = train_predict_transformer(series_train, series_test, args.forecast_horizon)
    elif args.model == 'NHiTS-Local':
        predictions = train_predict_nhits_local(series_train, series_test, args.forecast_horizon)
    elif args.model == 'NHiTS-Global':
        predictions = train_predict_nhits_global(
            series_train, series_test, args.forecast_horizon, args.seed)
    else:
        raise "Model Not Found."

    predictions = series_scaler.inverse_transform(predictions)

    scores = evaluate(predictions, series['test'])
    return scores


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Experimental Python script.")
    parser.add_argument("--dataset", type=str, default='paloalto',
                        help="Dataset: shell, paloalto, boulder")
    parser.add_argument("--model", type=str, default='Baseline', help="")
    parser.add_argument("--forecast_horizon", type=int,
                        default=1, help="Forecast horizon value")
    parser.add_argument("--input_chunk_length", type=int,
                        default=30, help="Input chunk length value")
    parser.add_argument("--use_covariates", type=bool,
                        default=False, help="Use covariates value")
    parser.add_argument("--train_data", type=int,
                        default=600, help="Train data value")
    parser.add_argument("--test_data", type=int,
                        default=90, help="Test data value")
    parser.add_argument("--subset", type=int,
                        default=None, help="Train data value")
    parser.add_argument("--seed", type=int, default=42, help="Random Seed")
    args = parser.parse_args()

    output = {'args': vars(args), 'results': {}}
    for model in ["Baseline", "ARIMA", "Transformer", "NHiTS-Local", "NHiTS-Global"]:
        args.model = model
        print(model)
        scores = main(args)
        output["results"][model] = scores

    with open(f'results/{args.dataset}_comparison_{args.horizon}_{args.seed}.json', 'w') as f:
        json.dump(output, f, indent=4)
