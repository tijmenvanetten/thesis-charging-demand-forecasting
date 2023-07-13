from models import load_model
from models import train_predict, train_predict_global, train_predict_past_covariates, train_predict_past_covariates_global
import sys
import os
import logging

sys.path.insert(1, os.path.join(sys.path[0], '../../src'))


def train_predict_baseline(series_train, series_test, forecast_horizon):
    predictions = []
    for series_train_single, series_test_single in zip(series_train, series_test):
        model = load_model("Baseline", input_chunk_length=None,
                           output_chunk_length=None, encoder=None, seed=None)

        forecast = train_predict(model,
                                 series_train=series_train_single,
                                 series_test=series_test_single,
                                 horizon=forecast_horizon,
                                 train_split=None,
                                 )

        predictions.append(forecast)
    return predictions


def train_predict_arima(series_train, series_test, forecast_horizon):
    predictions = []
    for series_train_single, series_test_single in zip(series_train, series_test):
        model = load_model("ARIMA", input_chunk_length=None,
                           output_chunk_length=None, encoder=None, seed=None)

        forecast = train_predict(model,
                                 series_train=series_train_single,
                                 series_test=series_test_single,
                                 horizon=forecast_horizon,
                                 train_split=None,
                                 )

        predictions.append(forecast)
    return predictions


def train_predict_transformer(series_train, series_test, encoder, forecast_horizon, covariates_train=None, covariates_test=None):
    predictions = []
    for series_train_single, series_test_single in zip(series_train, series_test):
        model = load_model("Transformer", input_chunk_length=None,
                           output_chunk_length=None, encoder=encoder, seed=None)

        forecast = train_predict(model,
                                 series_train=series_train_single,
                                 series_test=series_test_single,
                                 horizon=forecast_horizon,
                                 train_split=0.7,
                                 )

        predictions.append(forecast)
    return predictions


def train_predict_nhits_local(series_train, series_test, encoder, forecast_horizon, covariates_train=None, covariates_test=None):
    predictions = []
    for series_train_single, series_test_single in zip(series_train, series_test):
        model = load_model("NHiTS", input_chunk_length=None,
                           output_chunk_length=None, encoder=encoder, seed=None)

        if covariates_train:
            forecast = train_predict_past_covariates(model,
                                                     series_train=series_train_single,
                                                     series_test=series_test_single,
                                                     covariates_train=covariates_train,
                                                     covariates_test=covariates_test,
                                                     horizon=forecast_horizon,
                                                     train_split=0.7,
                                                     )
        else:
            forecast = train_predict(model,
                                     series_train=series_train_single,
                                     series_test=series_test_single,
                                     horizon=forecast_horizon,
                                     train_split=0.7,
                                     )

        predictions.append(forecast)
    return predictions


def train_predict_nhits_global(series_train, series_test, encoder, forecast_horizon, covariates_train=None, covariates_test=None):
    model = load_model("NHiTS", input_chunk_length=None,
                       output_chunk_length=None, encoder=encoder, seed=None)

    if covariates_train:
        predictions = train_predict_past_covariates_global(
            model, series_train, series_test, covariates_train, covariates_test, forecast_horizon, train_split=0.7)
    else:
        predictions = train_predict_global(
            model, series_train, series_test, forecast_horizon, train_split=0.7)
    return predictions
