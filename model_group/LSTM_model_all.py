import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import root_mean_squared_error, r2_score
import random
import os

# Reproducibility 
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

# ======= Manual LSTM =======
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

def create_sequences(data, n_past):             #create sequences for LSTM, split data into sequences of length n_past
    X, y = [], []
    for i in range(n_past, len(data)):
        X.append(data[i - n_past:i, :-1])
        y.append(data[i, -1])
    return np.array(X), np.array(y)             #convert a time series into a supervised learning problem

def prepare_dataloader(data, n_past, batch_size=64):
    X, y = create_sequences(data, n_past)
    if len(X) < 6:
        return None, None, None
    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.float32).unsqueeze(-1)
    dataset = TensorDataset(X_tensor, y_tensor)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True), X_tensor, y_tensor

# ======= Main per-street training =======
def main():
    #load and preprocess dataset
    df = pd.read_csv('/home/weichen/AI/project/AI_Group11_Final_Project/total_dataset_final.csv')
    df['Air_quality'] = pd.to_numeric(df['Air_quality'], errors='coerce').fillna(df['Air_quality'].mean())
    #convert string-based categorical columns into integers.
    categorical_cols = ['Boro', 'weekday', 'Direction', 'street']
    encoders = {col: LabelEncoder().fit(df[col]) for col in categorical_cols}
    for col in categorical_cols:
        df[col] = encoders[col].transform(df[col])

    features = ['Hour', 'weekday', 'temperature', 'precipitation', 'rain', 'cloudcover', 'windspeed', 'Air_quality', 'demand']
    target = 'volumn'

    df['Boro_street'] = df['Boro'].astype(str) + "_" + df['street'].astype(str)
    grouped = df.groupby('Boro_street')

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_past = 6
    results = []

    for name, group in grouped:                     #train group by group
        boro_code, street_code = name.split('_')
        boro = encoders['Boro'].inverse_transform([int(boro_code)])[0]          #original borough name
        street = encoders['street'].inverse_transform([int(street_code)])[0]    #original street name
        print(f"\n=== Training for group: {name} ===")
        print(f"  Boro: {boro}, Street: {street}")
        
        if len(group) < n_past + 10:                #if the data is too small
            continue
        #normalize the data
        group_data = group[features + [target]].values
        scaler = StandardScaler()
        group_scaled = scaler.fit_transform(group_data)
        scaler_y = StandardScaler()
        group_scaled[:, -1:] = scaler_y.fit_transform(group_scaled[:, -1:])
        
        #80% training, 20% validation
        train_size = int(0.8 * len(group_scaled))
        train_data = group_scaled[:train_size]
        val_data = group_scaled[train_size - n_past:]
        
        #create training sequences
        train_loader, _, _ = prepare_dataloader(train_data, n_past)
        val_X, val_y = create_sequences(val_data, n_past)
        val_X_tensor = torch.tensor(val_X, dtype=torch.float32)
        val_y_tensor = torch.tensor(val_y, dtype=torch.float32).unsqueeze(-1)
        
        #initialize the model
        model = LSTMModel(input_size=len(features), hidden_size=64, num_layers=3, dropout=0.1)

        #train the model
        print("Training...")
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
            #print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")

        #validation
        model.eval()
        with torch.no_grad():
            val_preds = model(val_X_tensor.to(device)).cpu().numpy()
            val_labels = val_y_tensor.cpu().numpy()
            val_preds = scaler_y.inverse_transform(val_preds)
            val_labels = scaler_y.inverse_transform(val_labels)
            val_rmse = root_mean_squared_error(val_labels, val_preds)
            val_mae = np.mean(np.abs(val_labels - val_preds))
            val_r2 = r2_score(val_labels, val_preds)

        print(f"Final Validation RMSE: {val_rmse:.4f}, MAE: {val_mae:.4f}, R2: {val_r2:.4f}")
        results.append({'boro': boro, 'street': street, 'RMSE': val_rmse, 'MAE': val_mae, 'R2': val_r2})
        #save the model
        # Make sure the models directory exists
        os.makedirs("models_group_all", exist_ok=True)
        torch.save(model.state_dict(), f"./models_group_all/LSTM_model_{boro}_{street}.pth")


    # Save results to CSV
    results_df = pd.DataFrame(results)
    results_df.to_csv("LSTM_group_results_all.csv", index=False)
    print("\nResults saved to LSTM_group_results_all.csv")

if __name__ == "__main__":
    main()
