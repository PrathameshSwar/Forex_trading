import numpy as np

try:
    scaler_params = np.load("scaler_params.npy", allow_pickle=True)
    print("✅ Scaler Params Loaded:", scaler_params)
    print("Type of Scaler Params:", type(scaler_params))
except Exception as e:
    print(f"❌ Error loading scaler: {e}")
