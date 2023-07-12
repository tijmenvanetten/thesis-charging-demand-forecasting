# Experiment 2: Influence of Training Length

This experiment script performs time series forecasting using the NHiTS model on different datasets. The script trains the NHiTS model locally and globally with varying training lengths and evaluates the performance of the model using various metrics.

## Dataset
The script supports three datasets: Shell, Palo Alto, and Boulder. The dataset is specified using the `--dataset` command-line argument.

## Parameters
The script accepts the following command-line arguments:
- `--dataset`: Specifies the dataset to use (default: paloalto)
- `--forecast_horizon`: Specifies the forecast horizon value (default: 1)
- `--input_chunk_length`: Specifies the input chunk length value (default: 30)
- `--use_covariates`: Specifies whether to use covariates (default: False)
- `--train_data`: Specifies the train data value (default: 600)
- `--test_data`: Specifies the test data value (default: 90)
- `--train_length`: Specifies the train length value (default: 330)

## Main Function
The main function of the script is `main(args)`. It loads the dataset, scales the series data, trains and predicts using the NHiTS model locally and globally, evaluates the predictions, and returns the results.

## Execution
The script can be executed by running the following command:

```bash
python -m train_length.py --dataset paloalto --forecast_horizon 1 --input_chunk_length 30 --use_covariates False --train_data 600 --test_data 90 --train_length 330
```