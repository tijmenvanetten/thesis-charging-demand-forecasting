import pandas as pd
from darts import TimeSeries
import argparse 
from pathlib import Path


def concat_partitions(sessions_dir: str) -> pd.DataFrame:
    """Concatenate input partitions into one pandas DataFrame.

    Args:
        partitioned_input: A dictionary with partition ids as keys and load functions as values.

    Returns:
        Pandas DataFrame representing a concatenation of all loaded partitions.
    """
    result = pd.DataFrame()

    for filepath in Path(sessions_dir).glob('*.csv'):
        df = pd.read_csv(filepath)
        # concat with existing result
        result = pd.concat([result, df], ignore_index=True, sort=True)

    return result


def aggregate_sessions(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate session data on location type, location and daily basis.

    Args:
        merged_input: A dataframe with the processed session data.

    Returns:
        Multiple Pandas Series containing the daily delivered energy per location.
    """
    
    df = df.groupby(['location_sub_type', 'location_id', 'date'])['energy_delivered_kwh'].sum()
    return df


def split_groups(df: pd.DataFrame) -> tuple[pd.DataFrame]:
    return df.groupby('location_sub_type')


def preprocess_sessions(df: pd.DataFrame) -> pd.DataFrame:
    """Process session data.

    Args:
        merged_input: A dataframe with a concatenation of all loaded partitions.

    Returns:
        Pandas DataFrame representing the processed session data.
    """
    # Convert column names to lowercase
    df.columns= df.columns.str.lower()

    df['session_start'] = pd.to_datetime(df['session_start'])
    df['date'] = df['session_start'].dt.date

    return df

def load_timeseries_dataset(df: pd.DataFrame) -> TimeSeries:
    df_ts = TimeSeries.from_group_dataframe(
        df, 
        group_cols='location_id', 
        time_col='date', 
        value_cols='energy_delivered_kwh', 
        fill_missing_dates=True, 
        freq='D',
    )
    series = [series for series in df_ts if len(series) == 1035]

    series_stacked = series[0]
    for series in series[1:]:
        series_stacked = series_stacked.stack(series)

    return series_stacked

def load_session_data(source_dir, target_dir):
    # Merge Session Files into Single Dataframe
    sessions = concat_partitions(source_dir)

    # Preprocess Session Dataframe
    preprocessed_sessions = preprocess_sessions(sessions)

    # Aggregate 
    aggregated_sessions = aggregate_sessions(preprocessed_sessions)


    # Split into multiple dataframes based on station type
    groups = split_groups(aggregated_sessions)
    for group, group_df in groups:
        group_df.to_csv(target_dir + group.lower().replace(' ', '_') + "_sessions.csv")

    


if __name__ == "__main__":
    # parser = argparse.ArgumentParser(description="Argument Parser")
    
    # parser.add_argument(
    #     "data_dir",
    #     type=str,
    # )

    # args = parser.parse_args()
    sessions = load_session_data('data/01_raw/sessions/', 'data/03_model_input/')
    # print(list(Path('data/01_raw/sessions/').glob('*.csv')))