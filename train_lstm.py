import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import matplotlib.pyplot as plt
import sys

print("✅ Step 1: Loading preprocessed data...")
sys.stdout.flush()

# Load the preprocessed data
X_train = np.load("/Users/prathameshswar/Downloads/Forex/X_train.npy")
X_test = np.load("/Users/prathameshswar/Downloads/Forex/X_test.npy")
y_train = np.load("/Users/prathameshswar/Downloads/Forex/y_train.npy")
y_test = np.load("/Users/prathameshswar/Downloads/Forex/y_test.npy")


print(f"✅ Data loaded! X_train shape: {X_train.shape}, X_test shape: {X_test.shape}")
sys.stdout.flush()

# Build the LSTM model
model = Sequential([
    LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])),
    Dropout(0.2),
    
    LSTM(units=50, return_sequences=True),
    Dropout(0.2),
    
    LSTM(units=50),
    Dropout(0.2),
    
    Dense(units=1)  # Predict next day's closing price
])

model.compile(optimizer='adam', loss='mean_squared_error')

print("✅ Step 2: Model built successfully!")
sys.stdout.flush()

# Train the model
EPOCHS = 50
BATCH_SIZE = 32

print(f"✅ Step 3: Training the model for {EPOCHS} epochs...")
sys.stdout.flush()

history = model.fit(X_train, y_train, epochs=EPOCHS, batch_size=BATCH_SIZE, validation_data=(X_test, y_test))

# Save the trained model
model.save("forex_lstm_model.h5")
print("✅ Step 4: Model training complete! Saved as 'forex_lstm_model.h5'")
sys.stdout.flush()

# Plot training loss
plt.figure(figsize=(10, 5))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('LSTM Training Loss Over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()
