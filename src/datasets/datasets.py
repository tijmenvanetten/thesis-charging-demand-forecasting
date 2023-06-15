from .data_utils import CovariatesDatasetLoader, TargetDatasetLoader

class WeatherEcadDataset(CovariatesDatasetLoader):
    def __init__(self):
        super().__init__(
            path='C:/Users/tijmen.vanetten/Documents/emobility-vanetten/data/03_processed/weather_ecad.csv',
            time_col='date', 
            value_cols=['temp_max', 'temp_min', 'sunshine', 'precip'],
            freq='D'
        )


class ShellDataset(TargetDatasetLoader):
    # Loads target time series data from a CSV file and returns a list of TimeSeries objects.
    def __init__(self, *args, **kwargs):
        super().__init__(
            path = 'C:/Users/tijmen.vanetten/Documents/emobility-vanetten/data/03_processed/shell_dataset.csv',
            group_cols = 'station_id',
            time_col = 'date',
            value_cols = 'energy_delivered_kwh',
            freq = 'D',
            static_cols = None,
            *args, **kwargs
        )

class PaloAltoDataset(TargetDatasetLoader):
    # Loads target time series data from a CSV file and returns a list of TimeSeries objects.
    def __init__(self, *args, **kwargs):
        super().__init__(
            path = 'C:/Users/tijmen.vanetten/Documents/emobility-vanetten/data/03_processed/palo_alto_dataset.csv',
            group_cols = 'station_id',
            time_col = 'date',
            value_cols = 'energy_delivered_kwh',
            freq = 'D',
            static_cols = None,
            *args, **kwargs
        )

class BoulderDataset(TargetDatasetLoader):
    # Loads target time series data from a CSV file and returns a list of TimeSeries objects.
    def __init__(self, *args, **kwargs):
        super().__init__(
            path = 'C:/Users/tijmen.vanetten/Documents/emobility-vanetten/data/03_processed/boulder_dataset.csv',
            group_cols = 'station_id',
            time_col = 'date',
            value_cols = 'energy_delivered_kwh',
            freq = 'D',
            static_cols = None,
            *args, **kwargs
        )