import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_squared_error
import random

# Reproducibility
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

# ======= Manual LSTM =======
class ManualLSTMCell(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.x2h = nn.Linear(input_size, 4 * hidden_size)
        self.h2h = nn.Linear(hidden_size, 4 * hidden_size)

    def forward(self, x, h_prev, c_prev):
        gates = self.x2h(x) + self.h2h(h_prev)
        i_gate, f_gate, g_gate, o_gate = gates.chunk(4, dim=1)
        i_gate = torch.sigmoid(i_gate)
        f_gate = torch.sigmoid(f_gate)
        g_gate = torch.tanh(g_gate)
        o_gate = torch.sigmoid(o_gate)
        c_t = f_gate * c_prev + i_gate * g_gate
        h_t = o_gate * torch.tanh(c_t)
        return h_t, c_t

class ManualLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=1, dropout=0.0):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.cells = nn.ModuleList([
            ManualLSTMCell(input_size if i == 0 else hidden_size, hidden_size)
            for i in range(num_layers)
        ])
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        batch_size, seq_len, _ = x.size()
        h = [torch.zeros(batch_size, self.hidden_size, device=x.device) for _ in range(self.num_layers)]
        c = [torch.zeros(batch_size, self.hidden_size, device=x.device) for _ in range(self.num_layers)]

        for t in range(seq_len):
            x_t = x[:, t, :]
            for layer in range(self.num_layers):
                h[layer], c[layer] = self.cells[layer](x_t, h[layer], c[layer])
                x_t = self.dropout(h[layer])

        return self.fc(h[-1])

# ======= Utilities =======
def create_sequences(data, n_past):
    X, y = [], []
    for i in range(n_past, len(data)):
        X.append(data[i - n_past:i, :-1])
        y.append(data[i, -1])
    return np.array(X), np.array(y)

def prepare_dataloader(data, n_past, batch_size=64):
    X, y = create_sequences(data, n_past)
    if len(X) < 10:
        return None, None, None
    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.float32).unsqueeze(-1)
    dataset = TensorDataset(X_tensor, y_tensor)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True), X_tensor, y_tensor

def train_model(model, train_loader, val_data, scaler_y, device, epochs=10, lr=0.001):
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    model.to(device)
    val_X_tensor, val_y_tensor = val_data

    for epoch in range(epochs):
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            output = model(xb)
            loss = criterion(output, yb)
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            val_preds = model(val_X_tensor.to(device)).cpu().numpy()
            val_labels = val_y_tensor.cpu().numpy()
            val_preds = scaler_y.inverse_transform(val_preds)
            val_labels = scaler_y.inverse_transform(val_labels)
            val_loss = mean_squared_error(val_labels, val_preds)
            print(f"Epoch {epoch+1}/{epochs}, Validation MSE: {val_loss:.4f}")

# ======= Main per-street training =======
def main():
    # Load and preprocess dataset
    df = pd.read_csv('/home/weichen/AI/project/code/total_dataset_final.csv')
    df['Air_quality'] = pd.to_numeric(df['Air_quality'], errors='coerce').fillna(df['Air_quality'].mean())

    categorical_cols = ['Boro', 'weekday', 'Direction', 'street', 'fromst', 'tost']
    encoders = {col: LabelEncoder().fit(df[col]) for col in categorical_cols}
    for col in categorical_cols:
        df[col] = encoders[col].transform(df[col])

    features = ['Hour', 'weekday', 'temperature', 'precipitation', 'rain',
                'cloudcover', 'windspeed', 'Air_quality', 'demand']
    target = 'volumn'

    df['Boro_street'] = df['Boro'].astype(str) + "_" + df['street'].astype(str)
    grouped = df.groupby('Boro_street')

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_past = 6
    results = []

    for name, group in grouped:
        print(f"\n=== Training for group: {name} ===")
        if len(group) < n_past + 10:
            print("  Skipped due to insufficient data.")
            results.append({'Boro_street': name, 'Status': 'Skipped (too little data)', 'Val_MSE': None})
            continue

        group_data = group[features + [target]].values
        scaler = StandardScaler()
        group_scaled = scaler.fit_transform(group_data)
        scaler_y = StandardScaler()
        group_scaled[:, -1:] = scaler_y.fit_transform(group_scaled[:, -1:])

        train_size = int(0.8 * len(group_scaled))
        train_data = group_scaled[:train_size]
        val_data = group_scaled[train_size - n_past:]

        train_loader, _, _ = prepare_dataloader(train_data, n_past)
        if train_loader is None:
            print("  Skipped due to sequence shortage.")
            results.append({'Boro_street': name, 'Status': 'Skipped (bad sequences)', 'Val_MSE': None})
            continue

        val_X, val_y = create_sequences(val_data, n_past)
        val_X_tensor = torch.tensor(val_X, dtype=torch.float32)
        val_y_tensor = torch.tensor(val_y, dtype=torch.float32).unsqueeze(-1)

        model = ManualLSTM(input_size=len(features), hidden_size=64, num_layers=2, dropout=0.3)

        # Train the model and evaluate
        print("  Training...")
        model.to(device)
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

        for epoch in range(10):
            model.train()
            for xb, yb in train_loader:
                xb, yb = xb.to(device), yb.to(device)
                optimizer.zero_grad()
                output = model(xb)
                loss = criterion(output, yb)
                loss.backward()
                optimizer.step()

        # Validation
        model.eval()
        with torch.no_grad():
            val_preds = model(val_X_tensor.to(device)).cpu().numpy()
            val_labels = val_y_tensor.cpu().numpy()
            val_preds = scaler_y.inverse_transform(val_preds)
            val_labels = scaler_y.inverse_transform(val_labels)
            val_mse = mean_squared_error(val_labels, val_preds)

        print(f"  Final Validation MSE: {val_mse:.4f}")
        results.append({'Boro_street': name, 'Status': 'Trained', 'Val_MSE': val_mse})

    # Save results to CSV
    results_df = pd.DataFrame(results)
    results_df.to_csv("lstm_group_results.csv", index=False)
    print("\n✅ Results saved to lstm_group_results.csv")

if __name__ == "__main__":
    main()
