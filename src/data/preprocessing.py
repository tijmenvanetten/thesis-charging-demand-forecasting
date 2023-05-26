import pandas as pd
from pathlib import Path
from typing import List


def concat_partitions(sessions_dir: str, usecols: List[str] | str) -> pd.DataFrame:
    """Concatenate input partitions into one pandas DataFrame.

    Args:
        partitioned_input: A dictionary with partition ids as keys and load functions as values.

    Returns:
        Pandas DataFrame representing a concatenation of all loaded partitions.
    """
    files = Path(sessions_dir).glob('*.csv')
    result = pd.concat([pd.read_csv(file, usecols=usecols) for file in files], ignore_index=True, sort=True)
    return result


def transform_datetime(df: pd.DataFrame, date_col='session_start') -> pd.DataFrame:
    """Process session data.

    Args:
        merged_input: A dataframe with a concatenation of all loaded partitions.

    Returns:
        Pandas DataFrame representing the processed session data.
    """
    df[date_col] = pd.to_datetime(df[date_col])
    df['date'] = df[date_col].dt.date
    return df


def aggregate(df: pd.DataFrame, level: List[str] | str = ['location_id', 'date'], target_col: str = 'energy_delivered_kwh') -> pd.DataFrame:
    """Aggregate session data on location type, location and daily basis.

    Args:
        df: A dataframe with the session data.
        level: level at which the data is aggregated
        target_col: target dimension that will be aggregated

    Returns:
        Multiple Pandas Series containing the daily delivered energy per location.
    """
    
    df = df.groupby(level)[target_col].sum()
    return df.reset_index()


def load_charging_station_info(filepath: str, group_col: str, value_cols: List[str] | str):
    """Add external data about charging stations.

    Args:
        merged_input: A dataframe with a concatenation of all loaded partitions.

    Returns:
        Pandas DataFrame representing the processed session data.
    """
    df = pd.read_csv(filepath)
    df = df[[group_col] + value_cols]

    return df
