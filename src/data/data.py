import pandas as pd
from typing import List
from darts import TimeSeries

from darts.utils.missing_values import fill_missing_values


def load_target(path: str, group_cols: List[str] | str, time_col: str, value_cols: List[str] | str,  freq: str, static_cols: List[str] | str = None) ->  List[TimeSeries]:
    df = pd.read_csv(path)

    # Load time series per location (group_cols)
    df_ts = TimeSeries.from_group_dataframe(
        df, 
        group_cols=group_cols, 
        time_col=time_col, 
        value_cols=value_cols, 
        static_cols=static_cols,
        fill_missing_dates=False, 
        freq=freq,
    )

    return df_ts


def load_covariates(path: str, time_col: str, value_cols: List[str], freq: str) -> TimeSeries:
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
    train_idx, val_idx = int(len(series) * train_split), int(len(series) * val_split)
    train, val, test = series[:train_idx], series[train_idx:val_idx], series[val_idx:]
    return train, val, test
    

if __name__ == "__main__":
    target = load_target('../data/01_raw/ChargePoint Data CY20Q4.csv', group_cols='Station Name', time_col='date', value_cols='energy_delivered_kwh', static_cols=['num_evse'], freq='D')
    # print(target.static_covariates)
    # covariates = load_covariates('../data/03_model_input/weather.csv', time_col='datetime', value_cols=['tempmax','tempmin', 'snow', 'precip'], freq='D')
    print(target.components)
