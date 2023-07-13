from darts.utils.likelihood_models import GaussianLikelihood
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from darts.dataprocessing.transformers.scaler import Scaler
import torch 

import sys
import os 
sys.path.insert(1, os.path.join(sys.path[0], '../../src'))


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

def load_model(model, input_chunk_length, output_chunk_length, encoder=None, seed=42, **kwargs):
    """
    Wrapper for all models
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
        return load_tftmodel(input_chunk_length, output_chunk_length, encoder=None, seed=42, **kwargs)
    elif model == 'TCN':
        return load_tcnmodel(input_chunk_length, output_chunk_length, encoder=None, seed=42, **kwargs)
    elif model == 'NBEATS':
        return load_nbeatsmodel(input_chunk_length, output_chunk_length, encoder=None, seed=42, **kwargs)
    elif model == 'XGB':
        return load_xgbmodel(input_chunk_length, output_chunk_length, encoder=None, seed=42, **kwargs)
    elif model == 'ARIMA':
        return load_arimamodel(input_chunk_length, output_chunk_length, encoder=None, seed=42, **kwargs)
    elif model == 'Baseline':
        return load_naivemean(input_chunk_length, output_chunk_length, encoder=None, seed=42, **kwargs)
    elif model == 'DeepAR':
        return load_deepar(input_chunk_length, output_chunk_length, encoder=None, seed=42, **kwargs)
    elif model == 'NHiTS':
        return load_nhitsmodel(input_chunk_length, output_chunk_length, encoder=None, seed=42, **kwargs)
    else:
        raise ValueError("Could not find specified model.")


def load_nhitsmodel(input_chunk_length, output_chunk_length, encoder=None, seed=42):
    from darts.models import NHiTSModel

    return NHiTSModel(
        nr_epochs_val_period=1,
        input_chunk_length=input_chunk_length,
        output_chunk_length=output_chunk_length,
        random_state=seed,
        add_encoders=encoder,
        pl_trainer_kwargs={"callbacks": [load_earlystopper()], "log_every_n_steps": 1},
    )


def load_deepar(input_chunk_length, output_chunk_length, encoder=None, seed=42):
    from darts.models import RNNModel
    return RNNModel(
        model="LSTM",
        hidden_dim=25, 
        n_rnn_layers=1,
        dropout=0.0,
        batch_size=16,
        add_encoders=encoder,
        nr_epochs_val_period=1,
        input_chunk_length=30,
        output_chunk_length=output_chunk_length,
        random_state=0,
        likelihood=GaussianLikelihood(),
        pl_trainer_kwargs={"callbacks": [load_earlystopper()]}
    )

def load_tcnmodel(input_chunk_length, output_chunk_length, encoder=None, seed=42):
    from darts.models import TCNModel
    return TCNModel(
        input_chunk_length=input_chunk_length,
        output_chunk_length=output_chunk_length,
        batch_size=16,
        num_layers=1,
        dilation_base=2,
        kernel_size=4,
        num_filters=3,
        dropout=0.2,
        likelihood=None,
        loss_fn=torch.nn.MSELoss(),
        optimizer_kwargs={'lr': 0.005}, 
        nr_epochs_val_period=1,
        add_encoders=encoder,
        n_epochs=20,
       
        random_state=0,
        pl_trainer_kwargs={"callbacks": [load_earlystopper()], "log_every_n_steps": 1},
    )

def load_tftmodel(input_chunk_length, output_chunk_length, encoder=None, seed=42):
    from darts.models import TFTModel
    return TFTModel(
        hidden_size=256, 
        lstm_layers=2,
        num_attention_heads=1,
        dropout=0.3,
        batch_size=32,
        optimizer_kwargs={'lr': 2e-3}, 
        nr_epochs_val_period=1,
        input_chunk_length=input_chunk_length,
        output_chunk_length=output_chunk_length,
        random_state=0,
        likelihood=None,
        loss_fn=torch.nn.MSELoss(),
        pl_trainer_kwargs={"callbacks": [load_earlystopper()], "log_every_n_steps": 1},
        add_relative_index=True,
    )

def load_nbeatsmodel(input_chunk_length, output_chunk_length, encoder=None, seed=42):
    from darts.models import NBEATSModel
    return NBEATSModel(
        input_chunk_length=input_chunk_length,
        output_chunk_length=output_chunk_length,
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
        add_encoders=encoder,
        likelihood=None,
        loss_fn=torch.nn.MSELoss(),
        pl_trainer_kwargs={"callbacks": [load_earlystopper()], "log_every_n_steps": 1},
        model_name="nbeats_model",
    )

def load_xgbmodel(input_chunk_length, output_chunk_length, encoder=None, seed=42):
    from darts.models import XGBModel
    return XGBModel(
        lags=input_chunk_length,
        lags_past_covariates=input_chunk_length,
        output_chunk_length=output_chunk_length,
        add_encoders=encoder,
        use_static_covariates=True,
        random_state=seed,
    )

def load_arimamodel(input_chunk_length, output_chunk_length, encoder=None, seed=42, p=30, d=0, q=30):
    from darts.models import ARIMA
    return ARIMA(
        p=p,
        d=d,
        q=q
    )


def load_naivemean(input_chunk_length, output_chunk_length, encoder=None, seed=42):
    from darts.models import NaiveMean
    return NaiveMean()
