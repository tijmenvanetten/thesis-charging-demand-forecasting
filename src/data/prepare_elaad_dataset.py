import pandas as pd
from preprocessing import transform_datetime, aggregate


def prepare_elaad_dataset(source_dir, target_dir):
    df = pd.read_excel(source_dir, usecols=['Start Date', 'Energy (kWh)', 'Station Name'])
    df = df.rename(columns={'Station Name': 'station_id', 'Energy (kWh)': 'energy_delivered_kwh'})

    df = transform_datetime(df, date_col='Start Date')
    df = aggregate(df, level=['station_id', 'date'], target_col='energy_delivered_kwh')
    df = df.reset_index(drop=True)
    df.to_csv(target_dir)

if __name__ == "__main__":
    prepare_elaad_dataset('data/01_raw/palo_alto_raw.csv', 'data/03_processed/palo_alto_dataset.csv')
    