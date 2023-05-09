from data.data import load_target, load_covariates
from models.models import load_model
from features.clustering import cluster_series
from models.train import train_predict_global, train_predict_local
from models.eval import evaluate
from visualization.visualize import plot, plot_separate
from tqdm import tqdm

import argparse
import os
from typing import Dict

from datetime import datetime


def run(config: Dict) -> Dict:
    """
    runs experiment

    """
    # Load Data
    target_series = load_target('data/03_processed/on_forecourt_sessions.csv', group_cols='location_id',
                                time_col='date', value_cols='energy_delivered_kwh', static_cols=['num_evse'], freq='D')
    covariates = load_covariates('data/03_processed/weather_ecad.csv', time_col='date',
                                 value_cols=['temp_max', 'temp_min', 'sunshine', 'precip'], freq='D')

    # Cluster Time Series
    series_clusters = cluster_series(target_series, k=config['clusters'])

    if config['model'] == 'DeepAR' or config['model'] == 'TFTModel':
        method = 'global'
    else:
        method = 'local'

    # Traing and Evaluate per cluster
    predictions = []
    for cluster in tqdm(series_clusters):
        model = load_model(config)

        if method == 'global':
            forecast = train_predict_global(
                model=model,
                target_series=cluster,
                covariates=covariates,
                forecast_horizon=config['forecast_horizon'],
                split=config['split'],
                scaler=config['scaler'],
                num_samples=config['num_samples'],
                stride=config['stride'],
                retrain=config['retrain'],
                verbose=config['verbose'])
        else:
            forecast = train_predict_local(
                model=model,
                target_series=cluster,
                covariates=covariates,
                num_samples=config['num_samples'],
                forecast_horizon=config['forecast_horizon'],
                split=config['split'],
            )

        predictions.append(forecast)

    print("Finished training")
    # Log Metrics
    scores = evaluate(predictions, series_clusters)
    logging.info(f"results: {scores}")

    # Save Forecast plot
    figs = plot_separate(predictions, series_clusters)
    for location_id, fig in figs.items():
        fig.savefig(config['logdir'] + "plots/" + f"{location_id}.png")
    


if __name__ == "__main__":
    import logging

    logging.getLogger("lightning.pytorch.utilities.rank_zero").setLevel(logging.WARNING)

    parser = argparse.ArgumentParser(description="Argument Parser")

    # Data options
    parser.add_argument("--location_type", type=str, default='on_forecourt')
    parser.add_argument("--clusters", type=int, default=None)

    # Training options
    parser.add_argument("--model", type=str, default="Naive",
                        help="a string specifying the model (Naive, DeepAR, ARIMA)")
    parser.add_argument("--split", type=float, default=0.7)
    parser.add_argument("--forecast_horizon", type=int, default=1)
    parser.add_argument("--num_samples", type=int, default=1)

    # Global Model Options
    parser.add_argument("--input_chunk_length", type=int)
    parser.add_argument("--hidden_dim", type=int,
                        help="Size for feature maps for each hidden RNN layer")
    parser.add_argument("--n_rnn_layers", type=int)
    parser.add_argument("--dropout", type=float)
    parser.add_argument("--num_epochs", type=int)
    parser.add_argument("--training_length", type=int)
    parser.add_argument("--scaler", type=str)
    parser.add_argument("--stride", type=int)
    parser.add_argument("--retrain", type=bool)
    parser.add_argument("--verbose", type=bool)
    parser.add_argument("--batch_size", type=int)
    parser.add_argument("--lr", type=float)

   
    args = parser.parse_args()
    config = vars(args)

    start_time_str = datetime.today().strftime('%d-%m-%Y_%H-%M-%S')
    logdir = f"logs/{start_time_str}_{config['model']}/"
    os.mkdir(logdir)
    os.mkdir(logdir + "plots/") 

    logging.basicConfig(filename=logdir + "logs.log", encoding='utf-8', level=logging.INFO)


    config['logdir'] = logdir

    logging.info(f"Running Experiments, params: {config}")
    
    run(config)
