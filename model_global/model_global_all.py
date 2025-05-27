import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import root_mean_squared_error, r2_score
import random
import matplotlib.pyplot as plt
from tqdm import tqdm
import os

SEED = 420
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
            nn.Dropout(dropout),
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

def main():
    # Load and preprocess dataset
    base_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(base_dir)
    csv_path = os.path.join(project_root, 'total_dataset_final.csv')
    df = pd.read_csv(csv_path)
    # df = pd.read_csv('../total_dataset_final.csv')
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date').reset_index(drop=True)
    df['Air_quality'] = pd.to_numeric(df['Air_quality'], errors='coerce').fillna(df['Air_quality'].mean())
    categorical_cols = ['Boro', 'weekday', 'Direction', 'street']
    encoders = {col: LabelEncoder().fit(df[col]) for col in categorical_cols}
    for col in categorical_cols:
        df[col] = encoders[col].transform(df[col])

    features = ['Hour', 'weekday', 'temperature', 'precipitation', 'rain', 'cloudcover', 'windspeed', 'Air_quality', 'demand']
    n_past = 6
    X = df[features]
    y = df['volumn']
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    target_scaler = StandardScaler()
    y_scaled = target_scaler.fit_transform(y.values.reshape(-1, 1))

    # Create sequences for the whole dataset
    X_seq, y_seq = create_sequences(X_scaled, y_scaled, n_past)
    if len(X_seq) < 10:
        print("Not enough data for sequence modeling.")
        return

    # Split into train/test (random 80/20 split)
    X_train, X_test, y_train, y_test = train_test_split(
        X_seq, y_seq, test_size=0.2, random_state=42, shuffle=True
    )
    noise_std = 0.01  # Standard deviation of noise
    X_train += np.random.normal(0, noise_std, X_train.shape)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LSTMModel(input_size=len(features), hidden_size=128, num_layers=3, dropout=0.3)
    model.to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=0.5, patience=3
)

    train_loader = DataLoader(
        TensorDataset(torch.tensor(X_train, dtype=torch.float32).to(device),
                      torch.tensor(y_train, dtype=torch.float32).to(device)),
        batch_size=128, shuffle=True
    )
    train_losses = []
    best_loss = float('inf')
    patience = 5
    wait = 0

    print("Training...")
    for epoch in range(60):
        model.train()
        epoch_loss = 0
        num_batches = 0
        with tqdm(train_loader, desc=f"Epoch {epoch+1}/60") as pbar:
            for xb, yb in pbar:
                xb, yb = xb.to(device), yb.to(device)
                optimizer.zero_grad()
                output = model(xb)
                loss = criterion(output, yb)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
                num_batches += 1
                pbar.set_postfix({'loss': loss.item()})
        avg_loss = epoch_loss / num_batches
        train_losses.append(avg_loss)
        print(f"Epoch {epoch+1} Average Loss: {avg_loss:.4f}")
        scheduler.step(avg_loss)
        if avg_loss < best_loss:
            best_loss = avg_loss
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                print("Early stopping triggered.")
                break

    #save the model
    #Make sure the models directory exists
    os.makedirs("model_global_all", exist_ok=True)
    save_dir = os.path.join(base_dir)
    torch.save(model.state_dict(), os.path.join(save_dir, 'LSTM_model_global_all.pth'))
    
    # Plot loss curve
    plt.figure()
    plt.plot(range(1, len(train_losses)+1), train_losses, marker='o')
    plt.xlabel('Epoch')
    plt.ylabel('Training Loss')
    plt.title('Training Loss Curve (all features)')
    save_path = os.path.join(base_dir, 'training_loss_global_all.png')
    plt.savefig(save_path)
    plt.show()
    plt.close()
    # Validation
    model.eval()
    with torch.no_grad():
        X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
        y_pred = model(X_test_tensor).detach().cpu().numpy().flatten()
        y_pred = target_scaler.inverse_transform(y_pred.reshape(-1, 1)).flatten()
        y_true = target_scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()
        val_rmse = root_mean_squared_error(y_true, y_pred)
        val_mae = np.mean(np.abs(y_true - y_pred))
        val_r2 = r2_score(y_true, y_pred)
        
    # Save all true and predicted values to CSV
    pred_df = pd.DataFrame({
        'True': y_true,
        'Predicted': y_pred
    })
    pred_df = pd.DataFrame({'True': y_true, 'Predicted': y_pred})
    pred_df.to_csv(os.path.join(save_dir, 'LSTM_predictions_global_all.csv'), index=False)
    print("\nAll predictions saved to LSTM_predictions_global_all.csv")

    print(f"Final Validation RMSE: {val_rmse:.4f}, MAE: {val_mae:.4f}, R2: {val_r2:.4f}")

    # Save results to CSV
    results_df = pd.DataFrame([{'RMSE': val_rmse, 'MAE': val_mae, 'R2': val_r2}])
    results_df.to_csv(os.path.join(save_dir, 'LSTM_results_global_all.csv'), index=False)
    print("\nResults saved to LSTM_results_global_all.csv")

if __name__ == "__main__":
    main()
