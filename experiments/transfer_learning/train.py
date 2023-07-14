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
from features.encoders import past_datetime_encoder
from models import load_model
from models.train import train_predict_global



def main(args):
    series_dataset = load_dataset(args.dataset)
    series = series_dataset.load(subset=args.subset, train_length=args.train_data, test_length=args.test_data, na_threshold=0.1)

    # Scale series Data
    series_scaler = Scaler(MinMaxScaler())
    series_train = series_scaler.fit_transform(series['train'])
    series_test= series_scaler.transform(series['test'])

    if args.datetime_encoder:
        encoder = past_datetime_encoder
    else:
        encoder = None

    # Train Model
    
    nhits_model = load_model("NHiTS", args.input_chunk_length, args.forecast_horizon, encoder, args.seed)
    
    nhits_model.fit(
        series=series_train,
        val_series=series_test,
    )

    nhits_model.save(f"models/{args.dataset}_nhits_global_{args.seed}_{args.subset}.pt")


    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Experimental Python script.")
    parser.add_argument("--dataset", type=str, default='shell', help="Dataset: shell, paloalto, boulder")
    parser.add_argument("--forecast_horizon", type=int, default=1, help="Forecast horizon value")
    parser.add_argument("--input_chunk_length", type=int, default=30, help="Input chunk length value")
    parser.add_argument("--use_covariates", type=bool, default=False, help="Use covariates value")
    parser.add_argument("--datetime_encoder", action="store_true", help="Encode datetime features")
    parser.add_argument("--subset", type=int, default=None, help="Subset of Time-series to use")
    parser.add_argument("--train_data", type=int, default=600, help="Train data value")
    parser.add_argument("--test_data", type=int, default=90, help="Test data value")
    parser.add_argument("--seed", type=int, default=42, help="Random Seed")
    args = parser.parse_args()

    print(args)

    main(args)

