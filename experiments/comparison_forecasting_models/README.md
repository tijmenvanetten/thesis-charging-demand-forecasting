# Experiment 1: Comparison of Forecasting Models

This experiment script performs time series forecasting using the NHiTS model on different datasets. The script trains the NHiTS model locally and globally with varying training lengths and evaluates the performance of the model using various metrics.

## Dataset
The script supports three datasets: Shell, Palo Alto, and Boulder. The dataset is specified using the `--dataset` command-line argument.

## Model
The script supports 5 models: Baseline, Transformer, ARIMA, NHiTS-Local, NHiTS-Global. The model is specified using the `--model` command-line argument.

## Parameters
The script accepts the following command-line arguments:
- `--dataset`: The dataset to use for the experiment.
- `--model`: The model to use for the experiment. 
- `--forecast_horizon`: The forecast horizon to use for the experiment. 
- `--input_chunk_length`: The input chunk length to use for the experiment. 
- `--use_covariates`: Whether to use covariates for the experiment. 
- `--use_encoder`: Whether to use an encoder for datetime features. 
- `--train_data`: Number of training points to use for the experiment. 
- `--test_data`: Number of test points to use for the experiment. 
- `--subset`: The subset time-series of the dataset to use for the experiment. 
- `--seed`: The random seed to use for the experiment. 

## Execution
The script can be executed by running the following command:

```bash
python main.py --dataset <dataset> --model <model> --forecast_horizon <forecast_horizon> --input_chunk_length <input_chunk_length> --use_covariates <use_covariates> --train_data <train_data> --test_data <test_data> --subset <subset> --seed <seed>

```
