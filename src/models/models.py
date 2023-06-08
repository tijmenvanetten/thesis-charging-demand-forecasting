from darts.utils.likelihood_models import GaussianLikelihood
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from darts.dataprocessing.transformers.scaler import Scaler
import torch 

encoders = {
            'cyclic': {'future': ['day']},
            'datetime_attribute': {'future': ['day', 'weekday', 'dayofweek']},
            'position': {'past': ['relative'], 'future': ['relative']},
            'custom': {'past': [lambda idx: (idx.year - 1950) / 50]},
            'transformer': Scaler()
        }

def load_earlystopper():
    return EarlyStopping(
        monitor="val_loss",
        patience=10,
        min_delta=0.01,
        mode='min',
    )

def load_model(args):
    if args.model == 'TFT':
       return load_tftmodel(args)
    if args.model == 'TCN':
        return load_tcnmodel(args)
    if args.model == 'NBEATS':
        return load_nbeatsmodel(args)
    if args.model == 'XGB':
        return load_xgbmodel(args)
    if args.model == 'ARIMA':
        return load_arimamodel(args)
    if args.model == 'VARIMA':
        return load_varimamodel(args)
    if args.model == 'NaiveMean':
        return load_naivemean(args)
    if args.model == 'DeepAR':
        return load_deepar(args)
    else:
        raise "Could not find specified model"

def load_deepar(args):
    from darts.models import RNNModel
    return RNNModel(
        model="LSTM",
        hidden_dim=25, 
        n_rnn_layers=1,
        dropout=0.0,
        batch_size=16,
        # add_encoders=encoders,
        # optimizer_kwargs={'lr': 0.0007}, 
        nr_epochs_val_period=1,
        log_tensorboard=args.logdir,
        input_chunk_length=30,
        random_state=0,
        likelihood=GaussianLikelihood(),
        pl_trainer_kwargs={"callbacks": [load_earlystopper()]}
    )

def load_tcnmodel(args):
    from darts.models import TCNModel
    return TCNModel(
        input_chunk_length=30,
        output_chunk_length=args.forecast_horizon,
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
        add_encoders=encoders,
        # log_tensorboard=True,
        n_epochs=20,
       
        random_state=0,
        work_dir=args.logdir + "/darts_logs/",
        pl_trainer_kwargs={"callbacks": [load_earlystopper()], "log_every_n_steps": 1} if not args.retrain else None,
    )

def load_tftmodel(args):
    from darts.models import TFTModel
    return TFTModel(
        # hidden_size=256, 
        # lstm_layers=2,
        # num_attention_heads=1,
        # dropout=0.3,
        # batch_size=32,
        # optimizer_kwargs={'lr': 2e-3}, 
        nr_epochs_val_period=1,
        # log_tensorboard=True,
        work_dir=args.logdir + "/darts_logs/",
        input_chunk_length=30,
        output_chunk_length=args.forecast_horizon,
        random_state=0,
        likelihood=None,
        loss_fn=torch.nn.MSELoss(),
        pl_trainer_kwargs={"callbacks": [load_earlystopper()], "log_every_n_steps": 1},
        add_relative_index=True,
    )

def load_nbeatsmodel(args):
    from darts.models import NBEATSModel
    return NBEATSModel(
        input_chunk_length=30,
        output_chunk_length=args.forecast_horizon,
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
        add_encoders=encoders,
        likelihood=None,
        loss_fn=torch.nn.MSELoss(),
        pl_trainer_kwargs={"callbacks": [load_earlystopper()], "log_every_n_steps": 1} if not args.retrain else None,
        model_name="nbeats_model",
        work_dir=args.logdir + "/darts_logs/",
    )

def load_xgbmodel(args):
    from darts.models import XGBModel
    return XGBModel(
        lags=30,
        lags_past_covariates=30,
        output_chunk_length=args.forecast_horizon,
        add_encoders=encoders,
        use_static_covariates=True,
    )

def load_arimamodel(args):
    from darts.models import ARIMA
    return ARIMA(
        p=30,
        d=0,
        q=30
    )


def load_varimamodel(args):
    from darts.models import VARIMA
    return VARIMA(
        
    )

def load_naivemean(args):
    from darts.models import NaiveMean
    return NaiveMean()
