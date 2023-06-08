import pandas as pd
from typing import List
from darts import TimeSeries

from darts.utils.missing_values import fill_missing_values


def stack_timeseries(series_list: List[TimeSeries]) -> List[TimeSeries]:
    """
    Combines a list of TimeSeries into one aligned multivariate TimeSeries.

    Args:
        series_list (List[TimeSeries]): A list of TimeSeries objects to be stacked.

    Returns:
        List[TimeSeries]: A list of TimeSeries objects representing the stacked multivariate TimeSeries.

    Example:
        >>> series1 = TimeSeries(...)
        >>> series2 = TimeSeries(...)
        >>> stacked_series = stack_timeseries([series1, series2])
    """
    # Transform TimeSeries back to pd.Series with location_id as title
    series = [
        series.pd_series().rename(series.static_covariates['station_id'][0])
        for series in series_list
    ]

    # Transform into aligned DataFrame
    df = pd.concat(series, axis=1)
    df = df.reindex(pd.date_range(df.index[0], df.index[-1], freq='D'))
    df = df.fillna(0)

    # Transform back to multivariate TimeSeries
    timeseries = TimeSeries.from_dataframe(df)

    return timeseries


def load_target(
    path: str,
    group_cols: List[str] | str = 'station_id',
    time_col: str = 'date',
    value_cols: List[str] | str = 'energy_delivered_kwh',
    freq: str = 'D',
    static_cols: List[str] | str = None
) -> List[TimeSeries]:
    """
    Loads target time series data from a CSV file and returns a list of TimeSeries objects.

    Args:
        path (str): The path to the CSV file.
        group_cols (List[str] | str, optional): The names of the columns to group the data by. Defaults to 'station_id'.
        time_col (str, optional): The name of the column containing the time values. Defaults to 'date'.
        value_cols (List[str] | str, optional): The names of the columns containing the target values. Defaults to 'energy_delivered_kwh'.
        freq (str, optional): The frequency of the time series data. Defaults to 'D'.
        static_cols (List[str] | str, optional): The names of the columns containing static features. Defaults to None.

    Returns:
        List[TimeSeries]: A list of TimeSeries objects, each representing a group of target time series.

    Example:
        >>> target_path = 'target.csv'
        >>> target_cols = ['energy_delivered_kwh']
        >>> target_ts = load_target(target_path, value_cols=target_cols)
    """
    # Read the CSV file into a pandas DataFrame
    df = pd.read_csv(path)

    # Load time series per location (group_cols)
    df_ts = TimeSeries.from_group_dataframe(
        df=df,
        group_cols=group_cols,
        time_col=time_col,
        value_cols=value_cols,
        static_cols=static_cols,
        fill_missing_dates=True,
        freq=freq,
    )

    # Fill missing values in each time series
    df_ts = [fill_missing_values(series) for series in df_ts]

    return df_ts


def load_covariates(path: str, value_cols: List[str], time_col: str = 'date', freq: str = 'D') -> TimeSeries:
    """
    Loads covariate data from a CSV file and returns a TimeSeries object.

    Args:
        path (str): The path to the CSV file.
        value_cols (List[str]): The names of the columns containing the covariate values.
        time_col (str, optional): The name of the column containing the time values. Defaults to 'date'.
        freq (str, optional): The frequency of the time series data. Defaults to 'D'.

    Returns:
        TimeSeries: A TimeSeries object containing the loaded covariate data.

    Example:
        >>> covariate_path = 'covariates.csv'
        >>> covariate_cols = ['temperature', 'humidity']
        >>> covariate_ts = load_covariates(covariate_path, covariate_cols)
    """
    # Read the CSV file into a pandas DataFrame
    df = pd.read_csv(path)

    # Create a TimeSeries object from the DataFrame
    df_ts = TimeSeries.from_dataframe(
        df=df,
        time_col=time_col,
        value_cols=value_cols,
        fill_missing_dates=False,
        freq=freq,
    )
    return df_ts



def split_data(series, train_split, val_split):
    """
    Splits a series of data into training, validation, and testing sets based on the given split ratios.

    Args:
        series (list or numpy array): The input series of data.
        train_split (float): The ratio (0-1) of data to be allocated for training.
        val_split (float): The ratio (0-1) of data to be allocated for validation.

    Returns:
        tuple: A tuple containing three sets: (train, val, test)
            - train (list or numpy array): The training set.
            - val (list or numpy array): The validation set.
            - test (list or numpy array): The testing set.

    Example:
        >>> series = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        >>> train, val, test = split_data(series, 0.6, 0.2)
        >>> print(train)
        [1, 2, 3, 4, 5, 6]
        >>> print(val)
        [7, 8]
        >>> print(test)
        [9, 10]
    """
    # Split training data
    train_idx, val_idx = int(len(series) * train_split), int(len(series) * val_split)
    train, val, test = series[:train_idx], series[train_idx:val_idx], series[val_idx:]
    return train, val, test