import ccxt
import time
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler

# Constants
MODEL_FILENAME = "forex_lstm_model.h5"
SCALER_FILENAME = "scaler_params.npy"
SEQUENCE_LENGTH = 20
BALANCE = 1000  # Initial virtual balance in USD

# Load Model
try:
    model = tf.keras.models.load_model(MODEL_FILENAME)
    print("✅ Model Loaded Successfully!")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    exit()

# Initialize Exchange API
exchange = ccxt.binance()
symbol = 'EURUSDT'

# Load Scaler
try:
    scaler_params = np.load(SCALER_FILENAME)
    scaler = MinMaxScaler()
    scaler.min_, scaler.scale_ = scaler_params.flatten()
    print("✅ Scaler Loaded Successfully!")
except Exception as e:
    print(f"❌ Error loading scaler: {e}")
    exit()

# Price Tracking
price_window = []
position = None  # "BUY" or "SELL"

while True:
    try:
        ticker = exchange.fetch_ticker(symbol)
        live_price = ticker['last']
        print(f"🔹 Live Price for {symbol}: ${live_price}")
    except Exception as e:
        print(f"❌ Error fetching live price: {e}")
        continue
    
    # Update rolling window
    price_window.append(live_price)
    if len(price_window) > SEQUENCE_LENGTH:
        price_window.pop(0)
    
    print(f"📊 Collected {len(price_window)}/{SEQUENCE_LENGTH} prices...")
    
    if len(price_window) == SEQUENCE_LENGTH:
        try:
            # Normalize & reshape input
            input_data = np.array(price_window).reshape(-1, 1)
            input_data = scaler.transform(input_data)
            input_data = np.expand_dims(input_data, axis=0)

            # Predict
            predicted_price = model.predict(input_data)
            predicted_price = scaler.inverse_transform(predicted_price)[0][0]
            print(f"🛠 Raw Model Prediction: ${predicted_price:.5f}")

            # Trading Logic
            if predicted_price > live_price and position != "BUY":
                print("✅ Buy Signal! Buying now...")
                position = "BUY"
            elif predicted_price < live_price and position != "SELL":
                print("❌ Sell Signal! Selling now...")
                position = "SELL"
        except Exception as e:
            print(f"❌ Error during prediction: {e}")
    
    time.sleep(5)  # Fetch price every 5 sec
