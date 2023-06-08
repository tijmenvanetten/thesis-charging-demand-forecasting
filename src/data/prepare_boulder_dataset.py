import pandas as pd
from preprocessing import transform_datetime, aggregate


def prepare_boulder_dataset(source_dir, target_dir):
    df = pd.read_csv(source_dir, usecols=['Station_Name', 'Start_Date___Time', 'Energy__kWh_'])
    df = df.rename(columns={'Station_Name': 'station_id', 'Energy__kWh_': 'energy_delivered_kwh'})

    df = transform_datetime(df, date_col='Start_Date___Time')
    df = aggregate(df, level=['station_id', 'date'], target_col='energy_delivered_kwh')
    df = df.reset_index(drop=True)
    df.to_csv(target_dir)

if __name__ == "__main__":
    prepare_boulder_dataset('data/01_raw/Electric_Vehicle_Charging_Station_Energy_Consumption.csv', 'data/03_processed/boulder_dataset.csv')
    