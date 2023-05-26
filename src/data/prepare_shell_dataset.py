import pandas as pd
import argparse 
from pathlib import Path
from typing import List

from preprocessing import concat_partitions, transform_datetime, aggregate, load_charging_station_info


def prepare_shell_dataset(session_dir: str, charge_info_path: str, target_path: str):
    df = concat_partitions(session_dir, usecols=['SESSION_START', 'LOCATION_ID', 'LOCATION_SUB_TYPE', 'ENERGY_DELIVERED_KWH'])
    df = transform_datetime(df, date_col='SESSION_START')
    df = aggregate(df, level=['LOCATION_SUB_TYPE', 'LOCATION_ID', 'date'], target_col='ENERGY_DELIVERED_KWH')

    df_station_info = load_charging_station_info(charge_info_path, 'LOCATION_ID', ['NUM_EVSE'])

    df = pd.merge(df, df_station_info, on='LOCATION_ID', how='left')
    df = df.loc[df['LOCATION_SUB_TYPE'] == 'On Forecourt']
    df = df.rename(columns={'LOCATION_ID': 'station_id'})
    df = df.reset_index(drop=True)
    df.to_csv(target_path)


if __name__ == "__main__":
    prepare_shell_dataset(session_dir='data/01_raw/shell_sessions', charge_info_path='data/01_raw/shell_stations/GB_shell_recharge_locations.csv', target_path='data/03_processed/shell_dataset.csv')
    