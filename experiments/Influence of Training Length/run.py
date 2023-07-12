import sys
import os
import json
import logging
import argparse
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from sklearn.preprocessing import MinMaxScaler

sys.path.insert(1, os.path.join(sys.path[0], '../../src'))
logging.getLogger("pytorch_lightning").setLevel(logging.WARNING)

from datasets import ShellDataset, PaloAltoDataset, BoulderDataset
from darts.dataprocessing.transformers.scaler import Scaler
from evaluation import evaluate
from features.encoders import past_datetime_encoder
from models import train_predict, train_predict_global
from darts.models import NHiTSModel

def load_nhitsmodel(input_chunk_length, forecast_horizon):
    return NHiTSModel(
        nr_epochs_val_period=1,
        input_chunk_length=input_chunk_length,
        output_chunk_length=forecast_horizon,
        random_state=0,
        add_encoders=past_datetime_encoder,
        pl_trainer_kwargs={"callbacks": [EarlyStopping(monitor="val_loss", patience=10, min_delta=0.01, mode='min')], "log_every_n_steps": 1},
    )

def main(args):
    # Set Experiment Parameters
    FORECAST_HORIZON = args.forecast_horizon
    INPUT_CHUNK_LENGTH = args.input_chunk_length
    USE_COVARIATES = args.use_covariates
    TRAIN_DATA = args.train_data
    TEST_DATA = args.test_data

    # Load Dataset
    if args.dataset == 'shell':
        series_dataset = ShellDataset()
    elif args.dataset == 'paloalto':
        series_dataset = PaloAltoDataset()
    elif args.dataset == 'boulder':
        series_dataset = BoulderDataset()
    else:
        raise Exception('Invalid dataset.')
    series = series_dataset.load(subset=None, train_length=TRAIN_DATA, test_length=TEST_DATA, na_threshold=0.1)

    # Decrease Training Length
    TRAIN_LENGTH = args.train_length
    series['train'] = [series_single[-TRAIN_LENGTH:] for series_single in series['train']]

    # Scale series Data
    series_scaler = Scaler(MinMaxScaler())
    series_train = series_scaler.fit_transform(series['train'])
    series_test= series_scaler.transform(series['test'])

    predictions = {}

    predictions_nhits = []
    for series_train_single, series_test_single in zip(series_train, series_test):
        model = load_nhitsmodel(INPUT_CHUNK_LENGTH, FORECAST_HORIZON)

        forecast = train_predict(model,
                            series_train=series_train_single,
                            series_test=series_test_single,
                            horizon=FORECAST_HORIZON,
                            train_split=0.7,
                            retrain=False)

        predictions_nhits.append(forecast)
    predictions_nhits = series_scaler.inverse_transform(predictions_nhits)
    predictions['NHiTS (Local)'] = predictions_nhits

    # Global Training
    nhits_model = load_nhitsmodel(INPUT_CHUNK_LENGTH, FORECAST_HORIZON)
    nhits_model_fit, predictions_nhits_global = train_predict_global(
                                                        model=nhits_model,
                                                        series_train=series_train,
                                                        series_test=series_test,
                                                        horizon=FORECAST_HORIZON,
                                                        train_split=0.7,
                                                        retrain=False
                                                    )

    predictions_nhits_global = series_scaler.inverse_transform(predictions_nhits_global)
    predictions['NHiTS (Global)'] = predictions_nhits_global

    # Evaluate
    result = []
    for model, model_predictions in predictions.items():
        scores = evaluate(model_predictions, series['test'])
        result.append({'model': model, 'scores': scores})
    return result

    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Experimental Python script.")
    parser.add_argument("--dataset", type=str, default='paloalto', help="Dataset: shell, paloalto, boulder")
    parser.add_argument("--forecast_horizon", type=int, default=1, help="Forecast horizon value")
    parser.add_argument("--input_chunk_length", type=int, default=30, help="Input chunk length value")
    parser.add_argument("--use_covariates", type=bool, default=False, help="Use covariates value")
    parser.add_argument("--train_data", type=int, default=600, help="Train data value")
    parser.add_argument("--test_data", type=int, default=90, help="Test data value")
    parser.add_argument("--train_length", type=int, default=330, help="Train length value")
    args = parser.parse_args()

    print(args)

    output = {'args': vars(args), 'results': {}}
    for train_length in [600, 570, 540, 510, 480, 450, 420, 390, 360, 330, 300, 270, 240, 210, 180, 150, 120]:
        args.train_length = train_length
        print(train_length)
        result = main(args)
        output["results"][train_length] = result

    with open(f'results/{args.dataset}_training_length.json', 'w') as f:
        json.dump(output, f, indent=4)

