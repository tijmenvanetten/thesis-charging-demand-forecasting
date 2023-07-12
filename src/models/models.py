from darts.utils.likelihood_models import GaussianLikelihood
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from darts.dataprocessing.transformers.scaler import Scaler
import torch 

import sys
import os 
sys.path.insert(1, os.path.join(sys.path[0], '../../src'))

from features.encoders import past_datetime_encoder

def load_earlystopper() -> EarlyStopping:
    """
    Loads and returns an instance of the EarlyStopping callback.

    Returns:
        EarlyStopping: An instance of the EarlyStopping callback.

    Example:
        >>> earlystopper = load_earlystopper()
    """
    return EarlyStopping(
        monitor="val_loss",
        patience=5,
        min_delta=0.01,
        mode='min',
    )

def load_model(model, model_args):
    """
    Loads and returns a specific model based on the provided arguments.

    Args:
        args: The arguments specifying the model to be loaded.

    Returns:
        The loaded model.

    Raises:
        ValueError: If the specified model is not found.

    Example:
        >>> model_args = ...
        >>> model = load_model(model_args)
    """
    if model == 'TFT':
        return load_tftmodel(model_args)
    elif model == 'TCN':
        return load_tcnmodel(model_args)
    elif model == 'NBEATS':
        return load_nbeatsmodel(model_args)
    elif model == 'XGB':
        return load_xgbmodel(model_args)
    elif model == 'ARIMA':
        return load_arimamodel(model_args)
    elif model == 'VARIMA':
        return load_varimamodel(model_args)
    elif model == 'NaiveMean':
        return load_naivemean(model_args)
    elif model == 'DeepAR':
        return load_deepar(model_args)
    elif model == 'NHiTS':
        return load_nhitsmodel(model_args)
    else:
        raise ValueError("Could not find specified model.")


def load_nhitsmodel(input_chunk_length, forecast_horizon, use_encoder=False):
    from darts.models import NHiTSModel

    return NHiTSModel(
        nr_epochs_val_period=1,
        input_chunk_length=input_chunk_length,
        output_chunk_length=forecast_horizon,
        random_state=1,
        add_encoders=past_datetime_encoder if use_encoder else None,
        pl_trainer_kwargs={"callbacks": [EarlyStopping(monitor="val_loss", patience=10, min_delta=0.01, mode='min')], "log_every_n_steps": 1},
    )


def load_deepar(model_args):
    from darts.models import RNNModel
    return RNNModel(
        model="LSTM",
        hidden_dim=25, 
        n_rnn_layers=1,
        dropout=0.0,
        batch_size=16,
        # add_encoders=encoders,
        nr_epochs_val_period=1,
        input_chunk_length=30,
        output_chunk_length=model_args.forecast_horizon,
        random_state=0,
        likelihood=GaussianLikelihood(),
        pl_trainer_kwargs={"callbacks": [load_earlystopper()]}
    )

def load_tcnmodel(model_args):
    from darts.models import TCNModel
    return TCNModel(
        input_chunk_length=30,
        output_chunk_length=model_args.forecast_horizon,
        batch_size=16,

        num_layers=1,
        dilation_base=2,
        kernel_size=4,
        num_filters=3,

        # weight_norm=True,
        dropout=0.2,

        likelihood=None,
        loss_fn=torch.nn.MSELoss(),

        optimizer_kwargs={'lr': 0.005}, 
        nr_epochs_val_period=1,
        add_encoders=past_datetime_encoder if model_args.use_encoder else None,
        n_epochs=20,
       
        random_state=0,
        pl_trainer_kwargs={"callbacks": [load_earlystopper()], "log_every_n_steps": 1},
    )

def load_tftmodel(model_args):
    from darts.models import TFTModel
    return TFTModel(
        hidden_size=256, 
        lstm_layers=2,
        num_attention_heads=1,
        dropout=0.3,
        batch_size=32,
        optimizer_kwargs={'lr': 2e-3}, 
        nr_epochs_val_period=1,
        input_chunk_length=30,
        output_chunk_length=model_args.forecast_horizon,
        random_state=0,
        likelihood=None,
        loss_fn=torch.nn.MSELoss(),
        pl_trainer_kwargs={"callbacks": [load_earlystopper()], "log_every_n_steps": 1},
        add_relative_index=True,
    )

def load_nbeatsmodel(model_args):
    from darts.models import NBEATSModel
    return NBEATSModel(
        input_chunk_length=30,
        output_chunk_length=model_args.forecast_horizon,
        batch_size=16,
        generic_architecture=False,
        nr_epochs_val_period=1,
        num_stacks=30,
        num_blocks=1,
        num_layers=4,
        layer_widths=256,
        expansion_coefficient_dim=5,
        trend_polynomial_degree=2,
        optimizer_kwargs={"lr": 2e-3},
        add_encoders=past_datetime_encoder if model_args.use_encoder else None,
        likelihood=None,
        loss_fn=torch.nn.MSELoss(),
        pl_trainer_kwargs={"callbacks": [load_earlystopper()], "log_every_n_steps": 1},
        model_name="nbeats_model",
    )

def load_xgbmodel(model_args):
    from darts.models import XGBModel
    return XGBModel(
        lags=30,
        lags_past_covariates=30,
        output_chunk_length=model_args.forecast_horizon,
        add_encoders=past_datetime_encoder if model_args.use_encoder else None,
        use_static_covariates=True,
    )

def load_arimamodel(model_args):
    from darts.models import ARIMA
    return ARIMA(
        p=30,
        d=0,
        q=30
    )


def load_varimamodel(model_args):
    from darts.models import VARIMA
    return VARIMA(
    )

def load_naivemean(model_args):
    from darts.models import NaiveMean
    return NaiveMean()
