import sys
import os
import json
import logging
import argparse

from sklearn.preprocessing import MinMaxScaler

sys.path.insert(1, os.path.join(sys.path[0], '../../src'))
logging.getLogger("pytorch_lightning").setLevel(logging.WARNING)

from datasets import load_dataset
from darts.dataprocessing.transformers.scaler import Scaler
from darts.models import NHiTSModel
from features.encoders import past_datetime_encoder
from models import load_model
from models.train import train_predict_global, predict_global
from evaluation import evaluate
from models.models import load_earlystopper


def main(args):
    series_dataset = load_dataset(args.dataset)
    series = series_dataset.load(subset=args.subset, train_length=args.train_data, test_length=args.test_data, na_threshold=0.1)

    # Scale series Data
    series_scaler = Scaler(MinMaxScaler())
    series_train = series_scaler.fit_transform(series['train'])
    series_test= series_scaler.transform(series['test'])

    # Load Pre-Trained Model
    model = NHiTSModel.load(args.model_path)
    
    # Reset early_stopper
    model.trainer_params['callbacks'] = [load_earlystopper()]

    # Retrain Model
    if args.retrain:
        predictions = train_predict_global(
            model, series_train, series_test, args.forecast_horizon, train_split=0.7)
    else:
        predictions = predict_global(
            model, series_train, series_test, args.forecast_horizon)

    # Scale back to original scale
    predictions = series_scaler.inverse_transform(predictions)

    # Evaluate
    scores = evaluate(predictions, series['test'])
    return scores

    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Experimental Python script.")
    parser.add_argument("--dataset", type=str, default='paloalto', help="Dataset: shell, paloalto, boulder")
    parser.add_argument("--forecast_horizon", type=int, default=1, help="Forecast horizon value")
    parser.add_argument("--model_path", type=str, required=True, help="Model Checkpoint to use")
    parser.add_argument("--subset", type=int, default=None, help="Subset of Time-series to use")
    parser.add_argument("--train_data", type=int, default=240, help="Train data value")
    parser.add_argument("--test_data", type=int, default=90, help="Test data value")
    parser.add_argument("--retrain", action='store_true', help="Retrain model")
    parser.add_argument("--seed", type=int, default=42, help="Random Seed")
    args = parser.parse_args()

    print(args)

    

    for dataset in ['paloalto', 'boulder']:
        output = {'args': vars(args), 'results': {}}

        for forecast_horizon in [1, 7, 30]:
            args.dataset = dataset
            args.forecast_horizon = forecast_horizon

            scores = main(args)

            output['results'][forecast_horizon] = {"NHiTS-Global-Pretrained": scores}

        
            with open(f'results/{args.dataset}_retrain={args.retrain}.json', 'w') as f:
                json.dump(output, f, indent=4)
            

