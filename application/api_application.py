import requests
import joblib

from datetime import datetime, timedelta
import pandas as pd
import torch
import os
from sklearn.preprocessing import StandardScaler, LabelEncoder
import numpy as np
from zoneinfo import ZoneInfo

#LSTM cell
class LSTMcell(torch.nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.x2h = torch.nn.Linear(input_size, 4 * hidden_size)
        self.h2h = torch.nn.Linear(hidden_size, 4 * hidden_size)

    def forward(self, x, h_prev, c_prev):
        gates = self.x2h(x) + self.h2h(h_prev)
        i_gate, f_gate, g_gate, o_gate = gates.chunk(4, dim=1)
        i_gate = torch.sigmoid(i_gate)
        f_gate = torch.sigmoid(f_gate)
        g_gate = torch.tanh(g_gate)
        o_gate = torch.sigmoid(o_gate)
        c = f_gate * c_prev + i_gate * g_gate
        h = o_gate * torch.tanh(c)
        return h, c

class LSTMModel(torch.nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=1, dropout=0.0):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = torch.nn.Dropout(dropout) if dropout > 0 else torch.nn.Identity()
        self.input_projection = torch.nn.Linear(input_size, input_size)
        self.cells = torch.nn.ModuleList([
            LSTMcell(input_size if i == 0 else hidden_size, hidden_size)
            for i in range(num_layers)
        ])
        self.fc = torch.nn.Sequential(
            torch.nn.Linear(hidden_size, 32),
            torch.nn.ReLU(),
            torch.nn.Linear(32, 1)
        )

    def forward(self, x):
        batch_size, seq_len, _ = x.size()
        x = self.input_projection(x)
        h = [torch.zeros(batch_size, self.hidden_size, device=x.device) for _ in range(self.num_layers)]
        c = [torch.zeros(batch_size, self.hidden_size, device=x.device) for _ in range(self.num_layers)]
        for t in range(seq_len):
            x_t = x[:, t, :]
            for layer in range(self.num_layers):
                h[layer], c[layer] = self.cells[layer](x_t, h[layer], c[layer])
                x_t = self.dropout(h[layer])
        return self.fc(h[-1])

def get_api_features(street_demand):
    headers = {
        'User-Agent': 'MyWeatherApp/1.0 tt121892185@gmail.com'
    }
    url = "https://weather.googleapis.com/v1/currentConditions:lookup?key=AIzaSyAFrnN5eet3dLywq5B118mR9hWV-F4B8Vo&location.latitude=40.7304&location.longitude=-74.0537"
    response = requests.get(url, headers=headers)
    if response.status_code == 200:
        data = response.json()
        temperature = data["temperature"]["degrees"]
        precipitation = data["precipitation"]["snowQpf"]["quantity"] + data["precipitation"]["qpf"]["quantity"]
        rain = data["precipitation"]["qpf"]["quantity"]
        cloudcover = data["cloudCover"]
        windspeed = data["wind"]["speed"]["value"]
    else:
        print("error code:", response.status_code)
        print("error message:", response.text)
        return None
    
    url = "https://api.airvisual.com/v2/city?city=New%20York%20City&state=New%20York&country=USA&key=7fc1f886-e778-41c3-83c6-1daf90fb85a9" # replace with your key
    response = requests.get(url, headers=headers)
    if response.status_code == 200:
        data = response.json()
        air_quality = data['data']['current']['pollution']['aqius']
    else:
        print("error code:", response.status_code)
        print("error message:", response.text)
        return None

    demand = street_demand  # If you have a way to estimate demand, set it here

    return temperature, precipitation, rain, cloudcover, windspeed, air_quality, demand
    return 12, 2, 0, 16, 0, 17, np.int64(3)

def main():
    # Load the dataset for encoders/scalers
    df = pd.read_csv("total_dataset_final.csv")
    features = ['Hour', 'weekday', 'temperature', 'precipitation', 'rain', 'cloudcover', 'windspeed', 'Air_quality', 'demand']

    # User input for street name
    street_name = input("Enter the street name: ").strip().upper()
    model_dir = "./model_street/models_street_all"
    model_path = os.path.join(model_dir, f"LSTM_model_{street_name}.pth")
    if not os.path.exists(model_path):
        print(f"Model for street '{street_name}' not found.")
        return

    # Instantiate model and load weights
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LSTMModel(input_size=len(features), hidden_size=64, num_layers=3, dropout=0.1)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    # Get current time
    current_time = datetime.now(ZoneInfo("America/New_York"))

    # Get API features (shared for all 6 hours for simplicity, or you can update per hour if you have hourly forecast)
    fliter_dp = df[df['street'] == street_name]
    street_demand =  fliter_dp.loc[0, 'demand']

    api_features = get_api_features(street_demand)
    #print(api_features)
    if api_features is None:
        print("Failed to get API features.")
        return
    scaler_x_path = os.path.join(model_dir, f"scaler_x_{street_name}.pkl")
    scaler_x = joblib.load(scaler_x_path)
    scaler_y_path = os.path.join(model_dir, f"scaler_y_{street_name}.pkl")
    scaler_y = joblib.load(scaler_y_path)
    print(f"\nPredicted traffic for street '{street_name}' for the next 6 hours:")
    for i in range(6):
        future_time = current_time + timedelta(hours=i)
        hour = future_time.hour
        weekday = 1 if future_time.weekday() < 5 else 0  # 'Y'->1, 'N'->0

        # Compose feature vector
        temperature, precipitation, rain, cloudcover, windspeed, air_quality, demand = api_features
        current_information = [hour, weekday, temperature, precipitation, rain, cloudcover, windspeed, air_quality, demand]
        current_information_df = pd.DataFrame([current_information], columns=features)
        current_information_scaled = scaler_x.transform(current_information_df)
        # Prepare input for LSTM: shape [1, seq_len, features]
        n_past = 1
        input_seq = np.tile(current_information_scaled, (n_past, 1))  # shape (6, features)
        input_tensor = torch.tensor(input_seq, dtype=torch.float32).unsqueeze(0)  # shape (1, 6, features)

        with torch.no_grad():
            pred_scaled = model(input_tensor.to(device)).cpu().numpy()
        pred = scaler_y.inverse_transform(pred_scaled)[0][0]  # Inverse transform to get the original scale
        print(f"Hour {hour:02d}:00 - Predicted volumn: {pred:.2f}")

if __name__ == "__main__":
    main()  