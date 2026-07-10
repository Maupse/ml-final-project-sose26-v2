from sklearn.metrics import root_mean_squared_error, mean_absolute_error, r2_score, mean_squared_error
def get_metrics(y_actual, y_pred):
    mse = mean_squared_error(y_actual, y_pred)
    rmse = root_mean_squared_error(y_actual, y_pred)
    mae = mean_absolute_error(y_actual, y_pred)
    r2 = r2_score(y_actual, y_pred)
    
    metrics = {
        "mse": mse,
        "rmse": rmse,
        "mae": mae,
        "r2": r2,
    }
    return metrics

def get_metrics_tuple(y_actual, y_pred):
    mse = mean_squared_error(y_actual, y_pred)
    rmse = root_mean_squared_error(y_actual, y_pred)
    mae = mean_absolute_error(y_actual, y_pred)
    r2 = r2_score(y_actual, y_pred)
    return mse, rmse, mae, r2