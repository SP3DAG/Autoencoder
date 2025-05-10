import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import numpy as np
import tensorflow as tf
from training.config import config

# Normalization constants (must match your training)
TIMESTAMP_SCALE = 1e10
LAT_SCALE = 90.0
LON_SCALE = 180.0

def denormalize(pred):
    timestamp = int(pred[0] * TIMESTAMP_SCALE)
    lat = pred[1] * LAT_SCALE
    lon = pred[2] * LON_SCALE
    return timestamp, lat, lon

# Load model
model = tf.keras.models.load_model(config["model_path"])

# Use part of training set as "test" set
X = np.load(config["train_images"])
y = np.load(config["train_labels"])

# Let's take the last 10% of training as a pseudo-test
split_index = int(len(X) * 0.9)
X_test, y_test = X[split_index:], y[split_index:]

# Evaluate
loss, mae = model.evaluate(X_test, y_test)
print(f"\nPseudo-Test Loss: {loss:.2f}, MAE: {mae:.2f}")

# Show sample predictions
print("\n--- Sample Predictions ---")
for i in range(min(5, len(X_test))):
    pred = model.predict(X_test[i:i+1])[0]
    true = y_test[i]

    pred_denorm = denormalize(pred)
    true_denorm = denormalize(true)

    print(f"\nSample {i+1}")
    print(f"Predicted → Timestamp: {pred_denorm[0]}, Lat: {pred_denorm[1]:.5f}, Lon: {pred_denorm[2]:.5f}")
    print(f"Actual    → Timestamp: {true_denorm[0]}, Lat: {true_denorm[1]:.5f}, Lon: {true_denorm[2]:.5f}")