import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np

def plot_with_error_rate(file_path, output_dir, title_suffix):
    df = pd.read_csv(file_path)
    os.makedirs(output_dir, exist_ok=True)
    df['boro_street'] = df['boro'].astype(str) + "_" + df['street'].astype(str)
    grouped = df.groupby('boro_street')
    for name, group in grouped:
        boro_name, street_name = name.split('_')
        subset = group.head(140)
        true_vals = subset['True'].values
        pred_vals = subset['Predicted'].values
        # Use np.divide to avoid divide by zero warning
        error_rate = np.empty_like(true_vals)
        error_rate[:] = np.nan
        np.divide(
            np.abs(true_vals - pred_vals), true_vals,
            out=error_rate, where=true_vals != 0
        )
        error_rate = error_rate * 100

        fig, ax1 = plt.subplots(figsize=(10, 5))
        ax1.plot(true_vals, label='True', color='blue', marker='o')
        ax1.plot(pred_vals, label='Predicted', color='orange', marker='x')
        ax1.set_xlabel('Time Index')
        ax1.set_ylabel('Traffic Volume')
        ax1.legend(loc='upper left')
        ax1.grid(True)

        ax2 = ax1.twinx()
        ax2.plot(error_rate, label='Error Rate', color='red', linestyle='--', marker='.')
        ax2.set_ylabel('Error Rate (%)')
        ax2.legend(loc='upper right')

        plt.title(f'Boro: {boro_name} Street: {street_name} ({title_suffix})')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"{boro_name}_{street_name.replace('/', '_')}_errorrate.png"))
        plt.close()

    print(f"Plots with error rate saved in folder: {output_dir}")

# Plot for all four prediction files
plot_with_error_rate('./model_group/LSTM_group_predict_all.csv', './model_group/group_error_all', 'using all features')
plot_with_error_rate('./model_group/LSTM_group_predict_demand.csv', './model_group/group_error_demand', 'using demand feature')
plot_with_error_rate('./model_group/LSTM_group_predict_historical.csv', './model_group/group_error_historical', 'using historical feature')
plot_with_error_rate('./model_group/LSTM_group_predict_weather.csv', './model_group/group_error_weather', 'using weather feature')