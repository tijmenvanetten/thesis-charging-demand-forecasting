# %%
import sys
import os
sys.path.insert(1, os.path.join(sys.path[0], '../../src'))

# %%
import logging
logging.getLogger("pytorch_lightning").setLevel(logging.WARNING)

# %%
from data import load_target, load_covariates

# %%
import torch
from darts.models import NBEATSModel
from darts.dataprocessing.transformers.scaler import Scaler
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from tqdm import tqdm
from optuna.integration import PyTorchLightningPruningCallback
import optuna
import numpy as np 
from sklearn.preprocessing import MaxAbsScaler
from darts.metrics import smape, mse
from darts.dataprocessing.transformers.scaler import Scaler
from darts.utils.likelihood_models import GaussianLikelihood
from darts import concatenate

# %%
def train_val_test_split(series, train_split: float, val_split: float):
    val_len = int(len(series) * train_split)
    test_len = int(len(series) * val_split)
    train, val, test = series[:val_len], series[val_len:test_len], series[test_len:]
    return train, val, test

# %%
# Load Data
target_series = load_target('../../data/03_processed/on_forecourt_sessions.csv', group_cols='location_id',
                            time_col='date', value_cols='energy_delivered_kwh', static_cols=['num_evse'], freq='D')
covariates = load_covariates('../../data/03_processed/weather_ecad.csv', time_col='date',
                                value_cols=['temp_max', 'temp_min', 'sunshine', 'precip'], freq='D')

target_series = [series for series in target_series if len(series) == 1035]
# Cluster Time Series
series = concatenate(target_series, axis=1)

TRAIN_SPLIT = 0.7
VAL_SPLIT = 0.85

train_series, val_series, test_series = train_val_test_split(series, TRAIN_SPLIT, VAL_SPLIT)

# scale target
target_scaler = Scaler(MaxAbsScaler())
train_series = target_scaler.fit_transform(train_series)
val_series = target_scaler.transform(val_series)
series_transformed = target_scaler.transform(series)


train_covariates, val_covariates, test_covariates = train_val_test_split(covariates, TRAIN_SPLIT, VAL_SPLIT)
# scale covariate
covariate_scaler = Scaler(MaxAbsScaler())
train_covariates = covariate_scaler.fit_transform(train_covariates)
val_covariates = covariate_scaler.transform(val_covariates)
covariates_transformed = covariate_scaler.transform(covariates)

train_val_series = concatenate([train_series, val_series])

# %%
# define objective function
def objective(trial):
    # select input and output chunk lengths
    in_len = trial.suggest_int("in_len", 12, 36)
    out_len = trial.suggest_int("out_len", 1, in_len-1)

    # Other hyperparameters
    num_stacks = trial.suggest_int("num_stacks", 1, 10)
    num_blocks = trial.suggest_int("num_blocks", 1, 5)
    num_layers = trial.suggest_int("num_layers", 1, 5)
    layer_widths = trial.suggest_int("layer_widths", 128, 512)
    generic_architecture= trial.suggest_categorical("generic_architecture", [False, True])
    lr = trial.suggest_float("lr", 5e-5, 1e-3, log=True)
    include_day = trial.suggest_categorical("day", [False, True])

    # throughout training we'll monitor the validation loss for both pruning and early stopping
    # pruner = PyTorchLightningPruningCallback(trial, monitor="val_loss")
    early_stopper = EarlyStopping("val_loss", min_delta=0.01, patience=3, verbose=True)

    pl_trainer_kwargs = {"callbacks": [ early_stopper]}
    num_workers = 0

    # optionally also add the (scaled) year value as a past covariate
    if include_day:
        encoders = {"datetime_attribute": {"past": ["day"]},
                    "transformer": Scaler()}
    else:
        encoders = None

    # reproducibility
    torch.manual_seed(42)

    # build the TCN model
    model = NBEATSModel(
        input_chunk_length=in_len,
        output_chunk_length=out_len,
        batch_size=32,
        n_epochs=100,
        nr_epochs_val_period=1,
        num_stacks=num_stacks,
        num_blocks=num_blocks,
        num_layers=num_layers,
        layer_widths=layer_widths,
        generic_architecture=generic_architecture,
        optimizer_kwargs={"lr": lr},
        add_encoders=encoders,
        likelihood=GaussianLikelihood(),
        pl_trainer_kwargs=pl_trainer_kwargs,
        model_name="nbeats_model",
        force_reset=True,
        save_checkpoints=True,
    )

    # train the model
    model.fit(
        series=train_series,
        past_covariates=covariates_transformed,
        val_series=val_series,
        val_past_covariates=covariates_transformed,
        num_loader_workers=num_workers,
    )

   
    # reload best model over course of training
    model = NBEATSModel.load_from_checkpoint("nbeats_model")

    # Evaluate how good it is on the validation set, using sMAPE
    # preds = model.predict(series=train, n=VAL_LEN)

    mses = model.backtest(
        train_val_series,
        start=val_series.start_time(),
        forecast_horizon=1,
        stride=1,
        last_points_only=False,
        retrain=False,
        verbose=True,
        metric=mse
    )
    mse_val = np.mean(mses)

    return mse_val if mse_val != np.nan else float("inf")


# for convenience, print some optimization trials information
def print_callback(study, trial):
    print(f"Current value: {trial.value}, Current params: {trial.params}")
    print(f"Best value: {study.best_value}, Best params: {study.best_trial.params}")


# optimize hyperparameters by minimizing the sMAPE on the validation set
study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=50, callbacks=[print_callback])

# %%
results = study.trials_dataframe()
results[results['value'] == results['value'].min()]

# %%


# %%



