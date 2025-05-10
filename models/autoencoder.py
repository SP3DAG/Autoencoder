import tensorflow as tf
from tensorflow.keras import layers

def build_metadata_recovery_autoencoder(input_shape=(128, 128, 3), output_dim=2):
    inputs = layers.Input(shape=input_shape)

    # Encoder
    x = layers.Conv2D(32, 3, strides=2, padding="same", activation="relu")(inputs)
    x = layers.Conv2D(64, 3, strides=2, padding="same", activation="relu")(x)
    x = layers.Flatten()(x)
    x = layers.Dense(256, activation="relu")(x)
    latent = layers.Dense(128, activation="relu", name="latent_space")(x)

    # Decoder (predict metadata)
    x = layers.Dense(64, activation="relu")(latent)
    outputs = layers.Dense(output_dim, activation=None)(x)

    return tf.keras.Model(inputs, outputs, name="MetadataRecoveryAutoencoder")