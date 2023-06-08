import pandas as pd
from typing import List
from darts import TimeSeries

from darts.utils.missing_values import fill_missing_values


def stack_timeseries(series_list: List[TimeSeries]) -> List[TimeSeries]:
    """
    Combines list of timeseries into one aligned multivariate timeseries

    """
    # Transform TimeSeries back to pd.Series with location_id as title
    series = [series.pd_series().rename(series.static_covariates['station_id'][0])
              for series in series_list]

    # Transform into aligned dataframe
    df = pd.concat(series, axis=1)
    df = df.reindex(pd.date_range(df.index[0], df.index[-1], freq='D'))
    # df = df.dropna(thresh=int(len(series[0])*0.5), axis=1)
    df = df.fillna(0)

    # Transform back to multivariate timeseries
    timeseries = TimeSeries.from_dataframe(df)
    return timeseries


def load_target(path: str, group_cols: List[str] | str = 'station_id', time_col: str = 'date', value_cols: List[str] | str = 'energy_delivered_kwh',  freq: str = 'D', static_cols: List[str] | str = None) -> List[TimeSeries]:
    df = pd.read_csv(path)
    # Load time series per location (group_cols)
    df_ts = TimeSeries.from_group_dataframe(
        df,
        group_cols=group_cols,
        time_col=time_col,
        value_cols=value_cols,
        static_cols=static_cols,
        fill_missing_dates=True,
        freq=freq,
    )

    df_ts = [fill_missing_values(series) for series in df_ts]
    return df_ts


def load_covariates(path: str, value_cols: List[str], time_col: str = 'date', freq: str = 'D') -> TimeSeries:
    df = pd.read_csv(path)

    df_ts = TimeSeries.from_dataframe(
        df=df,
        time_col=time_col,
        value_cols=value_cols,
        fill_missing_dates=False,
        freq=freq,
    )
    return df_ts


def split_data(series, train_split, val_split):
    # split training data
    train_idx, val_idx = int(
        len(series) * train_split), int(len(series) * val_split)
    train, val, test = series[:train_idx], series[train_idx:val_idx], series[val_idx:]
    return train, val, test

def data_handler(dataset:str, use_static_cols=None):
    if dataset == 'palo_alto':
        path = 'C:/Users/tijmen.vanetten/Documents/emobility-vanetten/data/03_processed/palo_alto_dataset.csv'
        time_col = 'date'
        static_cols=None
        covariates = None
        freq = 'D'
    elif dataset == 'shell':
        path = 'C:/Users/tijmen.vanetten/Documents/emobility-vanetten/data/03_processed/shell_dataset.csv'
        # static_cols= ['num_evse'] if use_static_cols else None,
        covariates = load_covariates(
            path='data/03_processed/weather_ecad.csv',
            value_cols=['temp_max', 'temp_min', 'sunshine', 'precip'],
            freq='D'
        )
        static_cols = None
        time_col = 'date'
        freq = 'D'
    elif dataset == 'boulder':
        path = 'C:/Users/tijmen.vanetten/Documents/emobility-vanetten/data/03_processed/boulder_dataset.csv'
        time_col = 'date'
        static_cols=None
        covariates = None
        freq = 'D'
    elif dataset == 'shell_weekly':
        path = 'C:/Users/tijmen.vanetten/Documents/emobility-vanetten/data/03_processed/shell_dataset_weekly.csv'
        time_col = 'week'
        static_cols=None
        covariates = None
        freq = 'W-MON'
    else:
        raise "Dataset not found"

    target_series = load_target(
        path=path,
        time_col=time_col,
        static_cols=static_cols,
        freq=freq
    )
    dataset = {}
    dataset['target'] = target_series 
    dataset['covariates'] = covariates
    return dataset

if __name__ == "__main__":
    target = load_target('../data/01_raw/ChargePoint Data CY20Q4.csv', group_cols='Station Name',
                         time_col='date', value_cols='energy_delivered_kwh', static_cols=['num_evse'], freq='D')
    # print(target.static_covariates)
    # covariates = load_covariates('../data/03_model_input/weather.csv', time_col='datetime', value_cols=['tempmax','tempmin', 'snow', 'precip'], freq='D')
    print(target.components)
