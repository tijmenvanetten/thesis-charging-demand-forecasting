import pandas as pd
from preprocessing import transform_datetime, aggregate


def prepare_palo_alto_dataset(source_dir, target_dir):
    df = pd.read_csv(source_dir, usecols=['Start Date', 'Energy (kWh)', 'Station Name'])
    df = transform_datetime(df, date_col='Start Date')
    df = aggregate(df, level=['Station Name', 'date'], target_col='Energy (kWh)')
    df = df.rename(columns={'station name': 'station_id', 'energy (kwh)': 'energy_delivered_kwh'})
    df = df.reset_index(drop=True)
    df.to_csv(target_dir)

if __name__ == "__main__":
    prepare_palo_alto_dataset('data/01_raw/palo_alto_raw.csv', 'data/03_processed/palo_alto_dataset.csv')
    