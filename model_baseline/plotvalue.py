import pandas as pd
import matplotlib.pyplot as plt
import os

# Load CSV file street predictions
file_path = './model_baseline/linear_regression_predict_all.csv'
df = pd.read_csv(file_path)

# Create output directory for plots
output_dir = './model_baseline/linear_regression_plots_all'
os.makedirs(output_dir, exist_ok=True)

# Group by street
grouped = df.groupby('street')

# Plot for each street
for street_name, group in grouped:
    subset = group.head(140)
    plt.figure(figsize=(10, 5))
    plt.plot(subset['True'].values, label='True', color='blue', marker='o')
    plt.plot(subset['Predicted'].values, label='Predicted', color='orange', marker='x')
    
    plt.title(f'Street: {street_name} (using all features)')
    plt.xlabel('Time Index')
    plt.ylabel('Traffic Volume')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    
    # Save each figure
    plt.savefig(os.path.join(output_dir, f"{street_name.replace('/', '_')}.png"))
    plt.close()

print(f"Plots saved in folder: {output_dir}")

# Load CSV file street predictions
file_path = './model_baseline/linear_regression_predict_demand.csv'
df = pd.read_csv(file_path)

# Create output directory for plots
output_dir = './model_baseline/linear_regression_plots_demand'
os.makedirs(output_dir, exist_ok=True)

# Group by street
grouped = df.groupby('street')

# Plot for each street
for street_name, group in grouped:
    subset = group.head(140)
    plt.figure(figsize=(10, 5))
    plt.plot(subset['True'].values, label='True', color='blue', marker='o')
    plt.plot(subset['Predicted'].values, label='Predicted', color='orange', marker='x')
    
    plt.title(f'Street: {street_name} (using demand feature)')
    plt.xlabel('Time Index')
    plt.ylabel('Traffic Volume')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    
    # Save each figure
    plt.savefig(os.path.join(output_dir, f"{street_name.replace('/', '_')}.png"))
    plt.close()

print(f"Plots saved in folder: {output_dir}")

# Load CSV file street predictions
file_path = './model_baseline/linear_regression_predict_historical.csv'
df = pd.read_csv(file_path)

# Create output directory for plots
output_dir = './model_baseline/linear_regression_plots_historical'
os.makedirs(output_dir, exist_ok=True)

# Group by street
grouped = df.groupby('street')

# Plot for each street
for street_name, group in grouped:
    subset = group.head(140)
    plt.figure(figsize=(10, 5))
    plt.plot(subset['True'].values, label='True', color='blue', marker='o')
    plt.plot(subset['Predicted'].values, label='Predicted', color='orange', marker='x')
    
    plt.title(f'Street: {street_name} (using historical feature)')
    plt.xlabel('Time Index')
    plt.ylabel('Traffic Volume')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    
    # Save each figure
    plt.savefig(os.path.join(output_dir, f"{street_name.replace('/', '_')}.png"))
    plt.close()

print(f"Plots saved in folder: {output_dir}")

# Load CSV file street predictions
file_path = './model_baseline/linear_regression_predict_weather.csv'
df = pd.read_csv(file_path)

# Create output directory for plots
output_dir = './model_baseline/linear_regression_plots_weather'
os.makedirs(output_dir, exist_ok=True)

# Group by street
grouped = df.groupby('street')

# Plot for each street
for street_name, group in grouped:
    subset = group.head(140)
    plt.figure(figsize=(10, 5))
    plt.plot(subset['True'].values, label='True', color='blue', marker='o')
    plt.plot(subset['Predicted'].values, label='Predicted', color='orange', marker='x')
    
    plt.title(f'Street: {street_name} (using weather feature)')
    plt.xlabel('Time Index')
    plt.ylabel('Traffic Volume')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    
    # Save each figure
    plt.savefig(os.path.join(output_dir, f"{street_name.replace('/', '_')}.png"))
    plt.close()

print(f"Plots saved in folder: {output_dir}")