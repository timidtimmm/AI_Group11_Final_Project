import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

df = pd.read_csv("/home/weichen/AI/project/AI_Group11_Final_Project/total_dataset_final.csv")

# 處理 weekday
if 'weekday' in df.columns and df['weekday'].dtype == 'object':
    df['weekday'] = df['weekday'].map({'Y': 1, 'N': 0})

features = ['Hour', 'weekday', 'temperature', 'precipitation', 'rain', 'cloudcover', 'windspeed', 'demand', 'Air_quality']
target = 'volumn'

X = df[features]
y = df[target]

# 資料標準化
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 訓練、測試集 (80% train, 20% test, random split)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# 線性回歸
model = LinearRegression()
model.fit(X_train, y_train)

# 預測
y_pred = model.predict(X_test)

# 指標
r2 = r2_score(y_test, y_pred)
rmse = mean_squared_error(y_test, y_pred) ** 0.5
mae = mean_absolute_error(y_test, y_pred)

# 儲存結果
results = [("all_data", r2, rmse, mae)]
pd.DataFrame(results, columns=["dataset", "R2", "RMSE", "MAE"]).to_csv("linear_regression_result_global.csv", index=False)