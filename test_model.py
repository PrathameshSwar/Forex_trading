import tensorflow as tf
import numpy as np

# Load the trained model
model = tf.keras.models.load_model("forex_lstm_model.h5")

# Generate a dummy input (50 timesteps)
dummy_input = np.random.rand(1, 50, 1)

# Predict
prediction = model.predict(dummy_input)
print("✅ Model Test Prediction:", prediction)
