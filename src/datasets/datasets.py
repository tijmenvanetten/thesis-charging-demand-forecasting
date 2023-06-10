import pandas as pd
from darts import TimeSeries, concatenate
from darts.dataprocessing.transformers import MissingValuesFiller
from typing import List

class DatasetLoader():
    """
    Base class for loading datasets.
    """
    def __init__(self, path, group_cols, time_col, value_cols, freq, static_cols=None, subset=None):
        self.path = path
        self.group_cols = group_cols
        self.time_col = time_col
        self.value_cols = value_cols
        self.freq = freq
        self.static_cols = static_cols
        self.subset = subset


    def load_timeseries_list(self) -> List[TimeSeries]:
        # Read the CSV file into a pandas DataFrame
        df = pd.read_csv(self.path)

        # Load time series per location (group_cols)
        df_ts = TimeSeries.from_group_dataframe(
            df=df,
            group_cols=self.group_cols,
            time_col=self.time_col,
            value_cols=self.value_cols,
            static_cols=self.static_cols,
            fill_missing_dates=True,
            freq=self.freq,
        )
        return df_ts
    
    def _split_data(self, series: TimeSeries, split=0.8):
        return series.split_after(split)
    
    

class ShellDataset(DatasetLoader):
    # Loads target time series data from a CSV file and returns a list of TimeSeries objects.
    def __init__(self, subset=None):
        super().__init__(
            path = 'C:/Users/tijmen.vanetten/Documents/emobility-vanetten/data/03_processed/shell_dataset.csv',
            group_cols = 'station_id',
            time_col = 'date',
            value_cols = 'energy_delivered_kwh',
            freq = 'D',
            static_cols = None,
            subset=subset
        )

    def load_dataset(self, split=0.8):
        """
        Custom process function for shell dataset
        """
        series = self.load_timeseries_list()
        series = [s for s in series if len(s) == 1035][:self.subset]
        transformer = MissingValuesFiller()
        series = transformer.transform(series)
        series = concatenate(series, axis=1)
        train, test = self._split_data(series, split)
        return {'train': train, 'test': test}

class PaloAltoDataset(DatasetLoader):
    # Loads target time series data from a CSV file and returns a list of TimeSeries objects.
    def __init__(self):
        super().__init__(
            path = 'C:/Users/tijmen.vanetten/Documents/emobility-vanetten/data/03_processed/palo_alto_dataset.csv',
            group_cols = 'station_id',
            time_col = 'date',
            value_cols = 'energy_delivered_kwh',
            freq = 'D',
            static_cols = None
        )
    
    def load_dataset(self):
        raise NotImplementedError


class BoulderDataset(DatasetLoader):
    # Loads target time series data from a CSV file and returns a list of TimeSeries objects.
    def __init__(self):
        super().__init__(
            path = 'C:/Users/tijmen.vanetten/Documents/emobility-vanetten/data/03_processed/boulder_dataset.csv',
            group_cols = 'station_id',
            time_col = 'date',
            value_cols = 'energy_delivered_kwh',
            freq = 'D',
            static_cols = None
        )

    def load_dataset(self):
        raise NotImplementedError
