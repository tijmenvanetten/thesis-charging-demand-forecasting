import seaborn as sns
import matplotlib.pyplot as plt

def plot_time_series_predictions(predictions_dict):
    sns.set_style("whitegrid")
    colors = sns.color_palette("Set1", len(predictions_dict))
    line_styles = ['-', '--', ':']  # Different line styles for each model
    line_widths = [1.5, 2.0, 1.0]  # Different line widths for each model

    # Calculate the number of rows and columns for subplots
    num_plots = list(predictions_dict.values())[0].n_components

    # Iterate over the forecast horizons
    for horizon, forecast_dict in predictions_dict.items():
        # Create a figure and subplots
        fig, axs = plt.subplots(num_plots, 1, figsize=(20, 6 * num_plots))

        # Flatten the subplots array if necessary
        if num_plots == 1:
            axs = [axs]

        # Iterate over the models and their predictions
        for i, (model, predictions) in enumerate(forecast_dict.items()):
            # Generate x values
            x = range(len(predictions[0]))

            # Iterate over predictions and plot each time series separately
            for j in range(predictions.n_components):
                time_series = predictions.univariate_component(j)
                x = time_series.time_index
                # Plot the time series line with different styles
                axs[j].plot(x, time_series.values(), color=colors[i], linestyle=line_styles[i % len(line_styles)],
                            linewidth=line_widths[i % len(line_widths)], label=f"{model}")

                # Plot points on the line indicating the predictions
                axs[j].scatter(x, time_series.values(), color=colors[i], marker='o', s=5, label=None)

                # Set labels and title for each subplot
                axs[j].set_xlabel('Time')
                axs[j].set_ylabel('Daily Energy Delivery Demand (kWh)')
                axs[j].set_title(f'Charging Station Forecasts (Horizon: {horizon})')

                # Show legend for each subplot
                axs[j].legend()

        # Adjust spacing between subplots
        # plt.tight_layout()

        # Display the plot
        plt.show()