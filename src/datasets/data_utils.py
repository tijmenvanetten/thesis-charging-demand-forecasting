from typing import List
from pandas import DataFrame
from darts import TimeSeries
import pandas as pd
from darts.dataprocessing.transformers import MissingValuesFiller

from abc import ABC


def split_time_series_list(time_series_list, length, select_last_only):
    # Create an empty list to store the split subseries
    split_subseries_list = []

    # Iterate over each TimeSeries in the list
    for time_series in time_series_list:
        # Get the start and end timestamps of the current TimeSeries
        start_timestamp = time_series.start_time()
        end_timestamp = time_series.end_time()

        # Calculate the total number of data points in the TimeSeries
        total_length = len(time_series)

        # Calculate the number of subseries to generate based on the length
        num_subseries = total_length // length

        if select_last_only:
            split_subseries_list.append(time_series[-length:])

        else:
            # Iterate over each subseries
            for i in range(num_subseries):
                # Calculate the start and end indices for the current subseries
                subseries_start = i * length
                subseries_end = subseries_start + length

                # Extract the data for the current subseries
                subseries_data = time_series.pd_dataframe(
                ).iloc[subseries_start:subseries_end]

                # Check if the extracted subseries has the desired length
                if len(subseries_data) == length:
                    # Create a new TimeSeries object for the current subseries
                    subseries_obj = TimeSeries.from_dataframe(subseries_data)

                    # Append the current subseries object to the list
                    split_subseries_list.append(subseries_obj)

    return split_subseries_list


def filter_timeseries(timeseries_list: List[DataFrame], threshold: float) -> List[DataFrame]:
    """
    Filters a list of darts TimeSeries objects based on the fraction of NA values.

    Args:
        dart_list (List[DataFrame]): List of darts TimeSeries objects.
        threshold (float): Threshold for the fraction of NA values.

    Returns:
        List[DataFrame]: List of filtered TimeSeries objects where the fraction of NA values is below the threshold.
    """
    filtered_list = []

    for timeseries in timeseries_list:
        na_count = timeseries.pd_series().isna().sum()
        total_count = len(timeseries)
        na_fraction = na_count / total_count

        if na_fraction < threshold:
            filtered_list.append(timeseries)

    return filtered_list


class CovariatesDatasetLoader(ABC):
    """
    Base class for loading datasets.
    """

    def __init__(self, path, time_col, value_cols, freq):
        self.path = path
        self.time_col = time_col
        self.value_cols = value_cols
        self.freq = freq

    def _split_data(self, series: TimeSeries, split=0.8):
        return series.split_after(split)


    def load(self) -> List[TimeSeries]:
        # Read the CSV file into a pandas DataFrame
        df = pd.read_csv(self.path)

        # Load time series per location (group_cols)
        covariates = TimeSeries.from_dataframe(
            df=df,
            time_col=self.time_col,
            value_cols=self.value_cols,
            fill_missing_dates=True,
            freq=self.freq,
        )
        return covariates
    
    def from_target_series(self, train_targets, test_targets):
        covariates = self.load()
        train_covariates, test_covariates = [], []
        for train_target_single, test_target_single in zip(train_targets, test_targets):
            train_covariates.append(covariates[train_target_single.time_index])
            test_covariates.append(covariates[test_target_single.time_index])
        return {'train': train_covariates, 'test': test_covariates}


class TargetDatasetLoader(ABC):
    """
    Base class for loading datasets.
    """

    def __init__(self, path, group_cols, time_col, value_cols, freq, static_cols=None):
        self.path = path
        self.group_cols = group_cols
        self.time_col = time_col
        self.value_cols = value_cols
        self.freq = freq
        self.static_cols = static_cols

    def load_timeseries_list(self) -> List[TimeSeries]:
        # Read the CSV file into a pandas DataFrame
        df = pd.read_csv(self.path)

        # Load time series per location (group_cols)
        df_ts = TimeSeries.from_group_dataframe(
            df=df,
            group_cols=self.group_cols,
            time_col=self.time_col,
            value_cols=self.value_cols,
            static_cols=self.static_cols,
            fill_missing_dates=True,
            freq=self.freq,
        )
        return df_ts

    
    def select_train_test(self, series, train_length, test_length):
        train, test = [], []
        for series_single in series:
            train_single = series_single[:train_length]
            test_single = series_single[train_length:]
            train.append(train_single)
            test.append(test_single)
        return train, test
    
    def filter_series_length(self, series, length):
        assert length > 0
        return [s[-length:] for s in series if len(s) >= length]
    
    def filter_series_na(self, series, na_threshold):
        assert na_threshold >= 0 and na_threshold <= 1
        return filter_timeseries(series, na_threshold)
    
    def fill_missing_values(self, series):
        transformer = MissingValuesFiller()
        return transformer.transform(series)
    
    def select_subset(self, series, subset):
        return series[:subset]


    def load(self, train_length, test_length, na_threshold=0.1, subset=None):
        # Load all time-series
        series = self.load_timeseries_list()
        print(
            f"Found {len(series)} time-series in dataset")
        
        # Filter out time series that are too short
        series = self.filter_series_length(series, train_length + test_length)
        print(
            f"Found {len(series)} time-series of length train_length + test_length = {train_length + test_length}")
        
        # Filter out time series with too many missing values
        series = self.filter_series_na(series, na_threshold)
        print(
            f"Found {len(series)} time-series with less than {na_threshold*100}% missing values")

        # Fill missing values
        series = self.fill_missing_values(series)
        print(
            f"Filled missing values in {len(series)} time-series")
        
        # Select subset if specified
        if subset is not None:
            series = self.select_subset(series, subset)
            print(
                f"Loaded {len(series)} time series each with length")
        
        # Split into train and test
        train, test = self.select_train_test(series, train_length, test_length)
        return {'train': train, 'test': test}
