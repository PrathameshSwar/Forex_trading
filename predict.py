import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

print("✅ Step 1: Loading trained LSTM model...")

# Load the trained model
model = tf.keras.models.load_model("forex_lstm_model.h5")
print("✅ Model loaded successfully!")

# Load test data
X_test = np.load("X_test.npy")
y_test = np.load("y_test.npy")
print(f"✅ Test data loaded! X_test shape: {X_test.shape}, y_test shape: {y_test.shape}")

# Make predictions
print("✅ Step 2: Making predictions...")
predictions = model.predict(X_test)

# Plot actual vs predicted values
plt.figure(figsize=(10, 5))
plt.plot(y_test, label="Actual Price", color='blue')
plt.plot(predictions, label="Predicted Price", color='red', linestyle='dashed')
plt.title("EUR/USD Price Prediction vs Actual")
plt.xlabel("Time")
plt.ylabel("Price")
plt.legend()
plt.show()

print("✅ Step 3: Prediction complete!")
