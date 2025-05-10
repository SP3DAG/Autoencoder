import numpy as np
import tensorflow as tf
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from models.autoencoder import build_metadata_recovery_autoencoder
from training.config import config

# Load data
X_train = np.load(config["train_images"])
y_train = np.load(config["train_labels"])

# Build model
model = build_metadata_recovery_autoencoder(
    input_shape=config["input_shape"],
    output_dim=config["output_dim"]
)

model.compile(optimizer="adam", loss="mse", metrics=["mae"])
model.summary()

# Callbacks
checkpoint = tf.keras.callbacks.ModelCheckpoint(config["model_path"], save_best_only=True)

# Train
model.fit(
    X_train, y_train,
    epochs=config["epochs"],
    batch_size=config["batch_size"],
    validation_split=0.1,
    callbacks=[checkpoint]
)