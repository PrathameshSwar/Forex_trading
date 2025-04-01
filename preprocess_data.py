import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import sys  # Import sys to force print output

print("✅ Step 1: Starting preprocessing script...")
sys.stdout.flush()

# Load data
try:
    df = pd.read_csv('/Users/prathameshswar/Downloads/Forex/EURUSD_historical_data.csv', index_col=0, parse_dates=True)
    print(f"✅ Step 2: Loaded CSV file successfully! Shape: {df.shape}")
    sys.stdout.flush()
except Exception as e:
    print(f"❌ Error: Could not load CSV file. {e}")
    sys.exit(1)  # Exit if file loading fails

# Selecting only the 'Close' price for LSTM
data = df[['Close']].values  
print(f"✅ Step 3: Selected 'Close' column. Data shape: {data.shape}")
sys.stdout.flush()

# Normalize data (scale between 0 and 1)
scaler = MinMaxScaler(feature_range=(0, 1))
data_scaled = scaler.fit_transform(data)
print(f"✅ Step 4: Data normalization complete. Sample: {data_scaled[:5]}")
sys.stdout.flush()

# Function to create sequences
def create_sequences(data, seq_length):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i+seq_length])
        y.append(data[i+seq_length])  # Next day's close price
    return np.array(X), np.array(y)

SEQ_LENGTH = 50  # Use past 50 days to predict the next day
X, y = create_sequences(data_scaled, SEQ_LENGTH)
print(f"✅ Step 5: Created sequences. X shape: {X.shape}, y shape: {y.shape}")
sys.stdout.flush()

# Split into training and testing sets (80% train, 20% test)
train_size = int(len(X) * 0.8)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

print(f"✅ Step 6: Data split. Train size: {train_size}, Test size: {len(X) - train_size}")
sys.stdout.flush()

# Save processed data
try:
    np.save("X_train.npy", X_train)
    np.save("X_test.npy", X_test)
    np.save("y_train.npy", y_train)
    np.save("y_test.npy", y_test)
    np.save("scaler.npy", scaler)  # Save the scaler for inverse transformation
    print("✅ Step 7: Saved .npy files successfully!")
    sys.stdout.flush()
except Exception as e:
    print(f"❌ Error: Could not save .npy files. {e}")
    sys.exit(1)

# Plot the scaled Close price
plt.figure(figsize=(10, 5))
plt.plot(data_scaled, label="Scaled Close Price")
plt.title("Scaled EUR/USD Close Price")
plt.xlabel("Days")
plt.ylabel("Scaled Price")
plt.legend()
plt.show()

print("✅ Step 8: Preprocessing completed successfully!")
sys.stdout.flush()
