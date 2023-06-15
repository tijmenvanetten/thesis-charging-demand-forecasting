import seaborn as sns
import matplotlib.pyplot as plt

def plot_time_series_predictions(predictions_dict, actual_values, horizon):
    sns.set_style("whitegrid")
    colors = sns.color_palette("Set1", len(predictions_dict))
    line_styles = ['-', '--', ':']  # Different line styles for each model
    line_width = 1.5  # Line width for all models

    # Calculate the number of rows and columns for subplots
    num_plots = len(list(predictions_dict.values())[0])

    # Create a figure and subplots
    fig, axs = plt.subplots(num_plots, 1, figsize=(20, 6 * num_plots))
    
    # Flatten the subplots array if necessary
    if num_plots == 1:
        axs = [axs]
    
    # Set the title for the entire figure
    fig.suptitle(f'Charging Station Forecasts (Horizon: {horizon})', fontsize=16)

    for j in range(len(actual_values)):
        time_series = actual_values[j]
        x = time_series.time_index
        # Plot the actual values for the time series
        axs[j].plot(x, time_series.values(), color='black', linestyle='-', label='Actual')


    for i, (model, predictions) in enumerate(predictions_dict.items()):
        # Iterate over the models and their predictions
        
        # Iterate over predictions and plot each time series separately
        for j in range(len(predictions)):
            time_series = predictions[j]
            x = time_series.time_index
            
            # Plot the time series line with the same line width and different styles
            axs[j].plot(x, time_series.values(), color=colors[i], linestyle=line_styles[i % len(line_styles)],
                        linewidth=line_width, label=f"{model}")

            # Plot points on the line indicating the predictions
            axs[j].scatter(x, time_series.values(), color=colors[i], marker='o', s=5, label=None)

            # Set labels for each subplot
            axs[j].set_xlabel('Time')
            axs[j].set_ylabel('Daily Energy Delivery Demand (kWh)')

            # Show legend for each subplot
            axs[j].legend()

    plt.tight_layout
    # Display the plot
    plt.show()
