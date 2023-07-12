import sys
import os
import logging
import argparse
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from sklearn.preprocessing import MinMaxScaler

sys.path.insert(1, os.path.join(sys.path[0], '../../src'))
logging.getLogger("pytorch_lightning").setLevel(logging.WARNING)

from datasets import ShellDataset
from darts.dataprocessing.transformers.scaler import Scaler
from evaluation import evaluate
from models import train_predict, train_predict_past_covariates, train_predict_global, train_predict_global_past_covariates
from models.models import load_model

def main(args):
    # Set Experiment Parameters
    FORECAST_HORIZON = args.forecast_horizon
    INPUT_CHUNK_LENGTH = args.input_chunk_length
    USE_COVARIATES = args.use_covariates
    TRAIN_DATA = args.train_data
    TEST_DATA = args.test_data

    # Load Dataset
    series_dataset = ShellDataset()
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
        model = load_model('nhits')

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
    nhits_model = load_model('nhits', )
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

    evaluate(predictions['NHiTS (Global)'], series['test'])

    # Results
    print(f"{USE_COVARIATES=},{FORECAST_HORIZON=},{INPUT_CHUNK_LENGTH=},{TRAIN_DATA=},{TEST_DATA=},{TRAIN_LENGTH=}, {series_dataset.__class__.__name__}, No. Series: {len(series['train'])}")
    for model, model_predictions in predictions.items():
        results = evaluate(model_predictions, series['test'])
        print(f"Model: {model}", results)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Experimental Python script.")
    parser.add_argument("--forecast_horizon", type=int, default=1, help="Forecast horizon value")
    parser.add_argument("--input_chunk_length", type=int, default=30, help="Input chunk length value")
    parser.add_argument("--use_covariates", type=bool, default=False, help="Use covariates value")
    parser.add_argument("--train_data", type=int, default=600, help="Train data value")
    parser.add_argument("--test_data", type=int, default=90, help="Test data value")
    parser.add_argument("--train_length", type=int, default=330, help="Train length value")
    args = parser.parse_args()

    main(args)
