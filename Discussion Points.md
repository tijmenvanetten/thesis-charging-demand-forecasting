Ideas to discuss:

TimeSeries selection, currently last n timesteps are selected of each charging station timeseries. Values with more NaN values than threshold (10%) are dropped. And a subset is selected, (how many locations to select?)
Works for Shell dataset, However no weather data yet for palo alto and boulder, can be done without but requires different model.

Generalisation to new TimeSeries, cross-validation across timeseries? Fine-Tune or not?

Only report accuracies for a single charging station that is used in other papers? Evaluate the influence of learning on other time series aswell

Further experiments to conduct?

Report if weather improves it, try with and without. 

Report mistakes in other papers. Put in related work/experiments/or discussion depending on where it fits. 

Introduction -> Related Work -> Method -> Experimental Setup -> Experiments -> Conclusion