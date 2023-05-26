import pandas as pd

def load_dataframe(filepath):
    df = pd.read_csv(filepath, skiprows=19)
    df.columns = df.columns.map(lambda x: x.strip().lower())
    df['date'] = pd.to_datetime(df['date'], format='%Y%m%d')
    return df


def preprocess():
    files = [
        # "data/01_raw/weather_ecad/CC_STAID001860.txt", 
        # "data/01_raw/weather_ecad/HU_STAID001860.txt", 
        # "data/01_raw/weather_ecad/QQ_STAID001860.txt",
        "data/01_raw/weather_ecad/RR_STAID001860.txt",
        # "data/01_raw/weather_ecad/SD_STAID001860.txt",
        "data/01_raw/weather_ecad/SS_STAID001860.txt",
        "data/01_raw/weather_ecad/TG_STAID001860.txt",
        "data/01_raw/weather_ecad/TN_STAID001860.txt",
        "data/01_raw/weather_ecad/TX_STAID001860.txt"
        ]

    df = pd.concat([load_dataframe(file) for file in files], axis=1)
    df = df.loc[:,~df.columns.duplicated()].copy()

    df = df[df['date'].notna()]

    df = df.rename(columns={
        "rr": "precip",
        "ss": "sunshine",
        "tg": "temp_avg",
        "tn": "temp_min",
        "tx": "temp_max"
    })
    df[['date', 'precip', 'sunshine', 'temp_avg', 'temp_min', 'temp_max']].to_csv("data/03_model_input/weather_ecad.csv")

if __name__ == "__main__":
    preprocess()