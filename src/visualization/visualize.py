import matplotlib.pyplot as plt

def plot(predictions, series_clusters):
    if len(predictions) == 1:
        rows = len(predictions[0].components)
    else:
        rows = len(predictions)
    cols=1

    fig, axs = plt.subplots(rows, cols, figsize=(14, 6 * rows))
    idx = 0
    for forecast_cluster, series_cluster in zip(predictions, series_clusters):
        components = forecast_cluster.components
        locations = forecast_cluster.static_covariates.station_id.values
        for component, location_id in zip(components, locations):
            forecast = forecast_cluster.univariate_component(component)
            actual = series_cluster.univariate_component(component)[forecast_cluster.time_index]

            forecast.plot(label="Forecast", ax=axs[idx])
            forecast.plot(low_quantile=0.05, high_quantile=0.95, label="5th and 95th percentiles", ax=axs[idx])
            actual.plot(label="Actual", ax=axs[idx])
            axs[idx].set_title(f"Forecast Evaluation, location id: {location_id}")
            axs[idx].legend(loc="upper left")
            idx += 1

    return fig, axs

def plot_separate(predictions, actuals):
    idx = 0
    figs = {}
    for prediction, actual in zip(predictions, actuals):
        for component in prediction.components:
            fig, axs = plt.subplots(1, 1, figsize=(14, 6))
            forecast = prediction.univariate_component(component)
            train, test = actual.univariate_component(component).split_after(prediction.start_time())

            # train.plot(label="Train", ax=axs)
            forecast.plot(label="Forecast", ax=axs)
            test.plot(label="Actual", ax=axs)
            axs.set_title(f"Forecast Evaluation - {idx=}")
            idx += 1
            figs[idx] = fig 
    return figs

def plot_series(series_list, cols=1):
    rows = round(len(series_list) / cols)
    fig, axs = plt.subplots(rows, cols, figsize=(14, 6 * rows))
    axs = axs.flatten()
    for idx, series in enumerate(series_list):
        series.plot(ax=axs[idx])
        axs[idx].set_title(f"Delivered Energy (kWH) over time - {idx=}")
    fig.tight_layout()
    return fig, axs

if __name__ == "__main__":
    from data.data import load_target
    target_series = load_target(
        path='data/03_processed/shell_dataset_weekly.csv',
        time_col='week',
        freq='W'
    )
    for series in target_series:
        plot()