import numpy as np

# Load the scaler object
scaler = np.load("scaler_params.npy", allow_pickle=True).item()

# Extract min_ and scale_
min_ = scaler.min_
scale_ = scaler.scale_

# Save only the required values
np.save("scaler_params.npy", np.array([min_, scale_], dtype=np.float64))

print("✅ Scaler parameters re-saved correctly!")
