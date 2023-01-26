import pandas as pd
from pathlib import Path

from config import SESSION_DATA_SRC_DIR, SESSION_DATA_TARGET_DIR, LOCATION_TYPES, SESSION_DATA_TARGET_NAME


def main():
    df = pd.DataFrame()

    path = Path(SESSION_DATA_SRC_DIR)
    for file in path.iterdir():
        if file.is_file() & (file.suffix == '.csv'):
            df_tmp = pd.read_csv(file)
            df = pd.concat([df, df_tmp])

    df['SESSION_STOP'] = pd.to_datetime(df['SESSION_STOP'], errors='coerce')
    df['SESSION_START'] = pd.to_datetime(df['SESSION_START'], errors='coerce')

    df = df[df['SESSION_STOP'].notna()]
    df = df[df['SESSION_START'].notna()]

    df['DATE'] = df['SESSION_STOP'].dt.date

    # drop the last date because it has not fully passed
    # df = df[df['DATE']!=df['DATE'].max()]

    # remove unwanted location subtypes
    df = df[df['LOCATION_SUB_TYPE'].isin(LOCATION_TYPES)]

    # save file
    df.to_csv(Path(SESSION_DATA_TARGET_DIR) / Path(SESSION_DATA_TARGET_NAME) )

if __name__ == "__main__":
    main()

