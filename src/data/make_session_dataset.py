import pandas as pd
import argparse 
from pathlib import Path
from typing import List


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
    return df.reset_index()


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


def load_charging_station_info(filepath: str, group_col: str, value_cols=List[str]):
    df = pd.read_csv(filepath)
    df.columns= df.columns.str.lower()

    df = df[[group_col] + value_cols]

    return df

def load_session_data(source_dir, recharge_data_path, target_dir):
    # Merge Session Files into Single Dataframe
    sessions = concat_partitions(source_dir)

    # Preprocess Session Dataframe
    preprocessed_sessions = preprocess_sessions(sessions)

    # Aggregate 
    agg_sessions = aggregate_sessions(preprocessed_sessions)

    # Load charging station information
    station_info = load_charging_station_info(recharge_data_path, 'location_id', ['num_evse'])

    # Merge Static Covariates with Target Series
    sessions_info = pd.merge(agg_sessions, station_info, on='location_id', how='left')
    print(sessions_info)

    # Split into multiple dataframes based on station type
    groups = sessions_info.groupby('location_sub_type')
    for group, group_df in groups:
        group_df.to_csv(target_dir + group.lower().replace(' ', '_') + "_sessions.csv")

    

if __name__ == "__main__":
    # parser = argparse.ArgumentParser(description="Argument Parser")
    
    # parser.add_argument(
    #     "data_dir",
    #     type=str,
    # )

    # args = parser.parse_args()
    load_session_data('01_raw/sessions/', "01_raw/stations/GB_shell_recharge_locations.csv", '03_model_input/')
    # print(list(Path('data/01_raw/sessions/').glob('*.csv')))

    # print(load_charging_station_info("data/01_raw/stations/GB_shell_recharge_locations.csv", 'location_id', ['num_evse']))