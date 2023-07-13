# Experiment 1: Comparison of Forecasting Models

This experiment script performs time series forecasting using the NHiTS model on different datasets. The script trains the NHiTS model locally and globally with varying training lengths and evaluates the performance of the model using various metrics.

## Dataset
The script supports three datasets: Shell, Palo Alto, and Boulder. The dataset is specified using the `--dataset` command-line argument.

## Parameters
The script accepts the following command-line arguments:
- `--dataset`: The dataset to use for the experiment. The default value is `shell`.
- `--model`: The model to use for the experiment. The default value is `nhits`.
- `--forecast_horizon`: The forecast horizon to use for the experiment. The default value is `1`.
- `--input_chunk_length`: The input chunk length to use for the experiment. The default value is `1`.
- `--use_covariates`: Whether to use covariates for the experiment. The default value is `False`.
- `--train_data`: The training data to use for the experiment. The default value is `train`.
- `--test_data`: The test data to use for the experiment. The default value is `test`.
- `--subset`: The subset of the dataset to use for the experiment. The default value is `all`.
- `--seed`: The random seed to use for the experiment. The default value is `0`.


## Main Function

## Execution
The script can be executed by running the following command:

```bash
python main.py --dataset <dataset> --model <model> --forecast_horizon <forecast_horizon> --input_chunk_length <input_chunk_length> --use_covariates <use_covariates> --train_data <train_data> --test_data <test_data> --subset <subset> --seed <seed>

```
