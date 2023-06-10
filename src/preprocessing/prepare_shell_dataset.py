import pandas as pd
import argparse 
from pathlib import Path
from typing import List

from preprocessing import concat_partitions, transform_datetime, aggregate, load_charging_station_info


def prepare_shell_dataset(session_dir: str, charge_info_path: str, target_path: str):
    df = concat_partitions(session_dir, usecols=['SESSION_START', 'LOCATION_ID', 'LOCATION_SUB_TYPE', 'ENERGY_DELIVERED_KWH'])
    df = df.rename(columns={'SESSION_START': 'session_start', 'LOCATION_ID': 'station_id', 'LOCATION_SUB_TYPE': 'location_sub_type', 'ENERGY_DELIVERED_KWH': 'energy_delivered_kwh'})
    df = df.loc[df['location_sub_type'] == 'On Forecourt']
    
    df = transform_datetime(df, date_col='session_start')
    df = aggregate(df, level=['station_id', 'week'], target_col='energy_delivered_kwh')
    df_station_info = load_charging_station_info(charge_info_path, 'LOCATION_ID', ['NUM_EVSE']).rename(columns={'NUM_EVSE': 'num_evse', 'LOCATION_ID': 'station_id'})

    df = pd.merge(df, df_station_info, on='station_id', how='left')

    df = df.reset_index(drop=True)
    df.to_csv(target_path)


if __name__ == "__main__":
    prepare_shell_dataset(session_dir='data/01_raw/shell_sessions', charge_info_path='data/01_raw/shell_stations/GB_shell_recharge_locations.csv', target_path='data/03_processed/shell_dataset_weekly.csv')