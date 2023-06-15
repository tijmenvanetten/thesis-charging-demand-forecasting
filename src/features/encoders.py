from darts.utils.timeseries_generation import holidays_timeseries
# import scaler from Darts
from darts.dataprocessing.transformers import Scaler

past_datetime_encoder = {
    'datetime_attribute': {'past': ['dayofweek']},
    'transformer': Scaler()
}

future_datetime_encoder = {
    'datetime_attribute': {'past': ['dayofweek']},
    'transformer': Scaler()
}

def get_past_holiday_encoder(time_index, country_code='GB'):
    return {
        'custom': {'past': [holidays_timeseries(time_index, country_code)]},
        'transformer': Scaler()
    }

def get_future_holiday_encoder(time_index, country_code='GB'):
    return {
        'custom': {'future': [holidays_timeseries(time_index, country_code)]},
        'transformer': Scaler()
    }
