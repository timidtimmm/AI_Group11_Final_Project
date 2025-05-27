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

r2, rmse, mae = evaluate_overall_metrics("model_street_test/LSTM_street_predict_all_test.csv", "model_street_test/LSTM_street_OverallValue_all_test.csv")
