import pandas as pd

# === Step 1: Read the CSV file ===
csv_file = 'total_dataset_final.csv'   # <-- Replace with your CSV filename
df = pd.read_csv(csv_file)

# === Step 2: Specify the column you want to count ===
target_column = 'street'  # <-- Replace with your column name

# === Step 3: Count distinct values ===
value_counts = df[target_column].value_counts()

# === Step 4: Display the result ===
print(value_counts)
