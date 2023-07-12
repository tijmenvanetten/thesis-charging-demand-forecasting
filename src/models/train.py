from darts import concatenate


def train_predict(model, series_train, series_test, horizon, train_split=None, retrain=False):
    series = concatenate([series_train, series_test])
    if train_split:
        series_train, series_val = series_train.split_after(train_split)
        model.fit(
            series=series_train,
            val_series=series_val,
        )
    else: 
       model.fit(
            series=series_train,
        )

    forecast = model.historical_forecasts(
        series=series,
        start=series_test.start_time(),
        forecast_horizon=horizon,
        retrain=retrain
    )
    return forecast


def train_predict_past_covariates(model, series_train, series_test, past_covariates_train, past_covariates_test, horizon, train_split=None, retrain=False):
    series = concatenate([series_train, series_test])
    past_covariates = concatenate([past_covariates_train, past_covariates_test])

    if train_split:
        series_train, series_val = series_train.split_after(train_split)
        past_covariates_train, past_covariates_val = past_covariates_train.split_after(train_split)

    model.fit(
        series=series_train,
        past_covariates=past_covariates_train,
        val_series=series_val,
        val_past_covariates=past_covariates_val
    )

    forecast = model.historical_forecasts(
        series=series,
        past_covariates=past_covariates,
        start=series_test.start_time(),
        forecast_horizon=horizon,
        retrain=retrain
    )
    return forecast

def train_predict_global(model, series_train, series_test, horizon, train_split=None, retrain=False):
    # create full series for historical forecast later
    series = [concatenate([series_train_single, series_test_single]) for series_train_single, series_test_single in zip(series_train, series_test)]

    if train_split:
        series_train_train, series_train_val = split_series_list(series_train, train_split)

    model.fit(
        series=series_train_train,
        val_series=series_train_val,
    )

    predictions = []
    for series_single, series_test_single in zip(series, series_test):
        forecast = model.historical_forecasts(
            series=series_single,
            start=series_test_single.start_time(),
            forecast_horizon=horizon,
            retrain=retrain
        )

        predictions.append(forecast)
    return model, predictions

def split_series_list(series_list, split):
    series_train_train, series_train_val = [], []
    for series_train_single in series_list:
        series_train_single, series_val_single = series_train_single.split_after(split)
        series_train_train.append(series_train_single)
        series_train_val.append(series_val_single)
    return series_train_train, series_train_val


def train_predict_global_past_covariates(model, series_train, series_test, past_covariates_train, past_covariates_test, horizon, train_split=None, retrain=False):
    # create full series for historical forecast later
    series = [concatenate([series_train_single, series_test_single]) for series_train_single, series_test_single in zip(series_train, series_test)]
    past_covariates = [concatenate([past_covariates_train_single, past_covariates_test_single]) for past_covariates_train_single, past_covariates_test_single in zip(past_covariates_train, past_covariates_test)]

    if train_split:
        series_train, series_val = split_series_list(series_train, train_split)
        past_covariates_train, past_covariates_val = split_series_list(past_covariates_train, train_split)
        

    model.fit(
        series=series_train,
        val_series=series_val,
        past_covariates=past_covariates_train,
        val_past_covariates=past_covariates_val
    )

    predictions = []
    for series_single, past_covariates_single, series_test_single in zip(series, past_covariates, series_test):
        forecast = model.historical_forecasts(
            series=series_single,
            past_covariates=past_covariates_single,
            start=series_test_single.start_time(),
            forecast_horizon=horizon,
            retrain=retrain
        )

        predictions.append(forecast)
    return model, predictions