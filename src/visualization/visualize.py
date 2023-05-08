import matplotlib.pyplot as plt

def visualize(predictions, series_clusters, model_name):
    rows, cols = len(predictions), 1
    fig, axs = plt.subplots(rows, cols, figsize=(14, 6 * rows))

    for i, (forecast, series) in zip(predictions, series_clusters):
        series.plot(ax=axs[i])
        forecast.plot(label=labels, ax=axs[i])
        axs[i].set_ylim(0, 500)
        axs[i].set_title("VARIMA model forecast")
        axs[i].legend(loc="upper left")
    return fig, axs