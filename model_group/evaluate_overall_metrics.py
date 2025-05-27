import pandas as pd
from sklearn.metrics import r2_score, root_mean_squared_error, mean_absolute_error

def evaluate_overall_metrics(filepath, output_csv):
    df = pd.read_csv(filepath)

    y_true = df['True'].values
    y_pred = df['Predicted'].values

    r2 = r2_score(y_true, y_pred)
    rmse = root_mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)

    metrics_df = pd.DataFrame([{
        "Overall_R2": r2,
        "Overall_RMSE": rmse,
        "Overall_MAE": mae
    }])
    metrics_df.to_csv(output_csv, index = False)

    return r2, rmse, mae

r2, rmse, mae = evaluate_overall_metrics("model_group/LSTM_group_predict_all.csv", "model_group/LSTM_group_OverallValue_all.csv")
r2, rmse, mae = evaluate_overall_metrics("model_group/LSTM_group_predict_demand.csv", "model_group/LSTM_group_OverallValue_demand.csv")
r2, rmse, mae = evaluate_overall_metrics("model_group/LSTM_group_predict_historical.csv", "model_group/LSTM_group_OverallValue_historical.csv")
r2, rmse, mae = evaluate_overall_metrics("model_group/LSTM_group_predict_weather.csv", "model_group/LSTM_group_OverallValue_weather.csv")