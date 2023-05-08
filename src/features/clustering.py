from typing import List, Dict
from darts import TimeSeries
import pandas as pd 
import numpy as np
from darts import concatenate

def stack_timeseries(series_list: List[TimeSeries]) -> List[TimeSeries]:
    """
    Combines list of timeseries into one aligned multivariate timeseries

    """
    # Transform TimeSeries back to pd.Series with location_id as title
    series = [series.pd_series().rename(series.static_covariates['location_id'][0])
              for series in series_list]

    # Transform into aligned dataframe
    df = pd.concat(series, axis=1)
    df = df.reindex(pd.date_range(df.index[0], df.index[-1], freq='D'))
    df = df.dropna(thresh=int(len(series[0])*0.5), axis=1)
    df = df.fillna(0)

    # Transform back to multivariate timeseries
    timeseries = TimeSeries.from_dataframe(df)
    return timeseries

def cluster_series(timeseries: List[TimeSeries], k: int) -> List[TimeSeries]:
    """
   Groups list of timeseries into clusters of timeseries of size k

    Args:
        k: number of clusters

    Returns:
        list of TimeSeries objects each representing a cluster 
    """

    # Temporariliy select subset
    timeseries = [series for series in timeseries if len(series) == 1035]

    if k == None:
        return timeseries
    if k == 1:
        timeseries = concatenate(timeseries, axis=1)
        return [timeseries]
    else:
        raise "Clustering method not implemented"