import sys
import os
sys.path.insert(1, os.path.join(sys.path[0], '../../src'))

import logging
import argparse
from sklearn.preprocessing import MinMaxScaler
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from darts.dataprocessing.transformers.scaler import Scaler
from datasets import BoulderDataset, ShellDataset, PaloAltoDataset, WeatherEcadDataset
from visualization import plot_time_series_predictions
from evaluation import evaluate
from models import train_predict, train_predict_global
from features.encoders import past_datetime_encoder
from darts.models.forecasting.baselines import NaiveMean
from darts.models import ARIMA, TransformerModel, NHiTSModel


logging.getLogger("pytorch_lightning").setLevel(logging.WARNING)


def load_dataset(dataset_name, train_length, test_length):
    if dataset_name == "BoulderDataset":
        dataset = BoulderDataset()
    elif dataset_name == "ShellDataset":
        dataset = ShellDataset()
    elif dataset_name == "PaloAltoDataset":
        dataset = PaloAltoDataset()
    else:
        raise ValueError("Invalid dataset name")

    series = dataset.load(
        subset=None, train_length=train_length, test_length=test_length, na_threshold=0.1
    )

    return series


def load_covariates(dataset_name, series_train, series_test):
    if dataset_name == "ShellDataset":
        weather_dataset = WeatherEcadDataset()
        covariates = weather_dataset.from_series(series_train, series_test)
        covariates_scaler = Scaler(MinMaxScaler())
        covariates_train = covariates_scaler.fit_transform(covariates["train"])
        covariates_test = covariates_scaler.transform(covariates["test"])
        return covariates_train, covariates_test
    else:
        return None, None


def scale_data(series_train, series_test):
    series_scaler = Scaler(MinMaxScaler())
    series_train_scaled = series_scaler.fit_transform(series_train)
    series_test_scaled = series_scaler.transform(series_test)

    return series_scaler, series_train_scaled, series_test_scaled


def train_baseline(series_train, series_test, horizon, retrain):
    model = NaiveMean()
    forecast = train_predict(
        model,
        series_train=series_train,
        series_test=series_test,
        horizon=horizon,
        retrain=retrain,
    )
    return forecast


def train_arima(series_train, series_test, horizon, retrain):
    model = ARIMA(p=len(series_train), d=0, q=len(series_train))
    forecast = train_predict(
        model,
        series_train=series_train,
        series_test=series_test,
        horizon=horizon,
        retrain=retrain,
    )
    return forecast


def train_transformer(series_train, series_test, horizon, retrain):
    model = TransformerModel(
        nr_epochs_val_period=1,
        nhead=8,
        num_encoder_layers=1,
        num_decoder_layers=1,
        dim_feedforward=128,
        input_chunk_length=len(series_train),
        output_chunk_length=horizon,
        random_state=0,
        pl_trainer_kwargs={
            "callbacks": [
                EarlyStopping(
                    monitor="val_loss",
                    patience=10,
                    min_delta=0.01,
                    mode="min",
                )
            ],
            "log_every_n_steps": 1,
        },
    )
    forecast = train_predict(
        model,
        series_train=series_train,
        series_test=series_test,
        horizon=horizon,
        train_split=0.7,
        retrain=retrain,
    )
    return forecast


def train_nhits(series_train, series_test, horizon, retrain):
    model = NHiTSModel(
        nr_epochs_val_period=1,
        input_chunk_length=len(series_train),
        output_chunk_length=horizon,
        random_state=0,
        pl_trainer_kwargs={
            "callbacks": [
                EarlyStopping(
                    monitor="val_loss",
                    patience=10,
                    min_delta=0.01,
                    mode="min",
                )
            ],
            "log_every_n_steps": 1,
        },
    )
    forecast = train_predict(
        model,
        series_train=series_train,
        series_test=series_test,
        horizon=horizon,
        train_split=0.7,
        retrain=retrain,
    )
    return forecast


def train_global_nhits(series_train, series_test, horizon, retrain):
    model = NHiTSModel(
        nr_epochs_val_period=1,
        input_chunk_length=len(series_train[0]),
        output_chunk_length=horizon,
        random_state=0,
        pl_trainer_kwargs={
            "callbacks": [
                EarlyStopping(
                    monitor="val_loss",
                    patience=10,
                    min_delta=0.01,
                    mode="min",
                )
            ],
            "log_every_n_steps": 1,
        },
    )
    model_fit, predictions_global = train_predict_global(
        model=model,
        series_train=series_train,
        series_test=series_test,
        horizon=horizon,
        train_split=0.7,
        retrain=retrain,
    )
    return predictions_global


def main(args):
    FORECAST_HORIZON = args.forecast_horizon
    INPUT_CHUNK_LENGTH = args.input_chunk_length
    USE_COVARIATES = args.use_covariates
    TRAIN_DATA = args.train_data
    TEST_DATA = args.test_data
    MODEL_NAME = args.model
    DATASET_NAME = args.dataset

    # Load Dataset
    series = load_dataset(DATASET_NAME, TRAIN_DATA, TEST_DATA)

    # Scale series Data
    series_scaler, series_train, series_test = scale_data(
        series["train"], series["test"]
    )

    # Load Covariates
    covariates_train, covariates_test = load_covariates(DATASET_NAME, series_train, series_test)

    predictions = {}

    # Baseline
    if MODEL_NAME == "Baseline":
        forecast = train_baseline(
            series_train,
            series_test,
            horizon=FORECAST_HORIZON,
            retrain=True,
        )
        predictions_baseline = series_scaler.inverse_transform([forecast])
        predictions["Baseline"] = predictions_baseline

        evaluate(predictions_baseline, series["test"])

    # Local Training
    if USE_COVARIATES and DATASET_NAME == "ShellDataset":
        encoder = past_datetime_encoder
    else:
        encoder = None

    if MODEL_NAME == "ARIMA":
        forecast = train_arima(
            series_train,
            series_test,
            horizon=FORECAST_HORIZON,
            retrain=False,
        )
        predictions_arima = series_scaler.inverse_transform([forecast])
        predictions["ARIMA"] = predictions_arima

        evaluate(predictions_arima, series["test"])

    if MODEL_NAME == "Transformer":
        forecast = train_transformer(
            series_train,
            series_test,
            horizon=FORECAST_HORIZON,
            retrain=False,
        )
        predictions_transformer = series_scaler.inverse_transform([forecast])
        predictions["Transformer"] = predictions_transformer

        evaluate(predictions_transformer, series["test"])

    if MODEL_NAME == "NHiTS (Local)":
        forecast = train_nhits(
            series_train,
            series_test,
            horizon=FORECAST_HORIZON,
            retrain=False,
        )
        predictions_nhits = series_scaler.inverse_transform([forecast])
        predictions["NHiTS (Local)"] = predictions_nhits

        evaluate(predictions_nhits, series["test"])

    # Global Training
    if MODEL_NAME == "NHiTS (Global)":
        predictions_global = train_global_nhits(
            series_train,
            series_test,
            horizon=FORECAST_HORIZON,
            retrain=False,
        )
        predictions_global = series_scaler.inverse_transform(predictions_global)
        predictions["NHiTS (Global)"] = predictions_global

        evaluate(predictions_global, series["test"])

    # Results
    print(
        f"{USE_COVARIATES=},{FORECAST_HORIZON=},{INPUT_CHUNK_LENGTH=},{TRAIN_DATA=},{TEST_DATA=}, {args.datset},No. Series:{len(series['test'])}"
    )
    for model, model_predictions in predictions.items():
        results = evaluate(model_predictions, series["test"])
        print(f"Model: {model}")
        print(results)

    # Visualization
    plot_time_series_predictions(predictions, series["test"], FORECAST_HORIZON)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the experiment.")
    parser.add_argument(
        "--forecast-horizon",
        type=int,
        default=30,
        help="Forecast horizon",
    )
    parser.add_argument(
        "--input-chunk-length",
        type=int,
        default=30,
        help="Input chunk length",
    )
    parser.add_argument(
        "--use-covariates",
        action="store_true",
        help="Use covariates",
    )
    parser.add_argument(
        "--train-data",
        type=int,
        default=240,
        help="Training data length",
    )
    parser.add_argument(
        "--test-data",
        type=int,
        default=90,
        help="Testing data length",
    )
    parser.add_argument(
        "--model",
        type=str,
        choices=["Baseline", "ARIMA", "Transformer", "NHiTS (Local)", "NHiTS (Global)"],
        default="Baseline",
        help="Model name",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        choices=["BoulderDataset", "ShellDataset", "PaloAltoDataset"],
        default="BoulderDataset",
        help="Dataset name",
    )

    args = parser.parse_args()
    main(args)
