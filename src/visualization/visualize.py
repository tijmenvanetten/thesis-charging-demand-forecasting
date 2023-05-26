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
        locations = forecast_cluster.static_covariates.location_id.values
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

def plot_separate(predictions, series_clusters):
    idx = 0
    figs = {}
    for forecast_cluster, series_cluster in zip(predictions, series_clusters):
        components = forecast_cluster.components
        locations = forecast_cluster.static_covariates.location_id.values
        for component, location_id in zip(components, locations):
            fig, axs = plt.subplots(1, 1, figsize=(14, 6))
            forecast = forecast_cluster.univariate_component(component)
            train, actual = series_cluster.univariate_component(component).split_after(forecast_cluster.start_time())

            train.plot(label="Train", ax=axs)
            forecast.plot(label="Forecast", ax=axs)
            actual.plot(label="Actual", ax=axs)
            axs.set_title(f"Forecast Evaluation, location id: {location_id}")
            idx += 1
            figs[location_id] = fig 
    return figs