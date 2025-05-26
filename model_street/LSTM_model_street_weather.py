import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import root_mean_squared_error, r2_score
import random
import os
import joblib

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)
#LSTM cell
class LSTMcell(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.x2h = nn.Linear(input_size, 4 * hidden_size)           #input to hidden layer
        self.h2h = nn.Linear(hidden_size, 4 * hidden_size)          #hidden to hidden layer

    def forward(self, x, h_prev, c_prev):
        gates = self.x2h(x) + self.h2h(h_prev)                      #linear transformation
        i_gate, f_gate, g_gate, o_gate = gates.chunk(4, dim=1)
        i_gate = torch.sigmoid(i_gate)              #input gate
        f_gate = torch.sigmoid(f_gate)              #forget gate
        g_gate = torch.tanh(g_gate)                 #cell gate
        o_gate = torch.sigmoid(o_gate)              #output gate 
        c = f_gate * c_prev + i_gate * g_gate       #cell state : c = f 。 c_prev + i 。 g
        h = o_gate * torch.tanh(c)                  #hidden state: taking the element-wise product of the output gate and the cell state
        return h, c                                 #pass to next cell
#LSTM model
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=1, dropout=0.0):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        
        self.input_projection = nn.Linear(input_size, input_size)   #input projection layer
        self.cells = nn.ModuleList([                                #define LSTM cells
            LSTMcell(input_size if i == 0 else hidden_size, hidden_size)
            for i in range(num_layers)
        ])
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        batch_size, seq_len, _ = x.size()
        x = self.input_projection(x)                   #input projection(learnable feature weighting)
        h = [torch.zeros(batch_size, self.hidden_size, device=x.device) for _ in range(self.num_layers)]
        c = [torch.zeros(batch_size, self.hidden_size, device=x.device) for _ in range(self.num_layers)]

        for t in range(seq_len):
            x_t = x[:, t, :]
            for layer in range(self.num_layers):
                h[layer], c[layer] = self.cells[layer](x_t, h[layer], c[layer])
                x_t = self.dropout(h[layer])

        return self.fc(h[-1])

def create_sequences(X, y, n_past):
    X_seq, y_seq = [], []
    for i in range(n_past, len(X)):
        X_seq.append(X[i-n_past:i])
        y_seq.append(y[i])
    return np.array(X_seq), np.array(y_seq)

# training by grouping by street and using only weather data
def main():
    #load and preprocess dataset
    df = pd.read_csv('total_dataset_final.csv')
    df['Air_quality'] = pd.to_numeric(df['Air_quality'], errors='coerce').fillna(df['Air_quality'].mean())
    #convert string-based categorical columns into integers.
    categorical_cols = ['Boro', 'weekday', 'Direction', 'street']
    encoders = {col: LabelEncoder().fit(df[col]) for col in categorical_cols}
    for col in categorical_cols:
        df[col] = encoders[col].transform(df[col])

    features = ['Hour', 'temperature', 'precipitation', 'rain', 'cloudcover', 'windspeed', 'Air_quality']

    n_past = 6
    X = df[features]
    y = df['volumn']
    #boro = df['Boro']
    street = df['street']
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    target_scaler = StandardScaler()
    y_scaled = target_scaler.fit_transform(y.values.reshape(-1, 1))

    df = pd.DataFrame(X_scaled, columns=features)
    df['volumn'] = y_scaled
    #df['Boro'] = boro.values
    df['street'] = street.values
    #df['Boro_street'] = df['Boro'].astype(str) + "_" + df['street'].astype(str)
    grouped = df.groupby('street')
    results = []

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for name, group in grouped:                     #train group by group
        #boro_code, street_code = name.split('_')
        #boro = encoders['Boro'].inverse_transform([int(boro_code)])[0]          #original borough name
        street_name = encoders['street'].inverse_transform([int(name)])[0]    #original street name
        print(f"\n=== Training for group: {name} ===")
        print(f"  Street: {street_name} ")
        X_group = group[features].values
        y_group = group['volumn'].values.reshape(-1, 1)
        X_seq, y_seq = create_sequences(X_group, y_group, n_past)
        if len(X_seq) < 10:                #if the data is too small
            continue
        X_train, X_test, y_train, y_test = train_test_split(X_seq, y_seq, test_size=0.2, random_state=42)

        #initialize the model
        model = LSTMModel(input_size=len(features), hidden_size=64, num_layers=3, dropout=0.1)

        #train the model
        print("Training...")
        model.to(device)
        criterion = nn.MSELoss()  # Huber Loss
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

        train_loader = DataLoader(
            TensorDataset(torch.tensor(X_train, dtype=torch.float32).to(device),
                        torch.tensor(y_train, dtype=torch.float32).to(device)),
            batch_size=64, shuffle=True
        )
        for epoch in range(20):
            model.train()
            for xb, yb in train_loader:
                xb, yb = xb.to(device), yb.to(device)
                optimizer.zero_grad()
                output = model(xb)
                loss = criterion(output, yb)
                loss.backward()
                optimizer.step()
            #print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")

        #validation
        model.eval()
        with torch.no_grad():
            X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
            y_pred = model(X_test_tensor).detach().cpu().numpy().flatten()
            y_pred = target_scaler.inverse_transform(y_pred.reshape(-1, 1)).flatten()
            y_true = target_scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()
            val_rmse = root_mean_squared_error(y_true, y_pred)
            val_mae = np.mean(np.abs(y_true - y_pred))
            val_r2 = r2_score(y_true, y_pred)

        print(f"Final Validation RMSE: {val_rmse:.4f}, MAE: {val_mae:.4f}, R2: {val_r2:.4f}")
        results.append({'street': street_name, 'RMSE': val_rmse, 'MAE': val_mae, 'R2': val_r2})
        #save the model
        #make sure the models directory exists and store the model and scaler values
        os.makedirs("./model_street/models_street_weather", exist_ok=True)
        torch.save(model.state_dict(), f"./model_street/models_street_weather/LSTM_model_{street_name}.pth")
        joblib.dump(scaler, f"./model_street/models_street_weather/scaler_x_{street_name}.pkl")
        joblib.dump(target_scaler, f"./model_street/models_street_weather/scaler_y_{street_name}.pkl")

    # Save results to CSV
    results_df = pd.DataFrame(results)
    results_df.to_csv("./model_street/LSTM_street_results_weather.csv", index=False)
    print("\nResults saved to LSTM_street_results_weather.csv")

if __name__ == "__main__":
    main()
