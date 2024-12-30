import numpy as np
import os
import random
import tensorflow as tf

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2

# Ensure reproducibility
random_seed = 42
np.random.seed(random_seed)
tf.random.set_seed(random_seed)
random.seed(random_seed)
os.environ["TF_DETERMINISTIC_OPS"] = "1"


class Autoencoder:
    def __init__(self, scaled_ratios_data, encoding_dim=2, random_state=42):
        """
        Initializes the Autoencoder class.
        :param scaled_ratios_data: Input data as a dictionary of DataFrames or arrays.
        :param encoding_dim: Number of dimensions for the reduced representation.
        :param random_state: Seed for reproducibility.
        """
        self.reconstructed_data = None

        self.random_state = random_state
        self.scaled_ratios_data = scaled_ratios_data

        self.autoencoder_result = {}
        self.encoding_dim = encoding_dim
        self.encoder = None
        self.autoencoder_model = None
        self.reduced_data = None

    def build_autoencoder(self):
        """
        Builds the autoencoder model with controlled randomness.
        """
        tf.random.set_seed(self.random_state)

        input_data = Input(shape=(7,))  # Adjust to accept the number of features for each time point

        # Encoder
        encoded = Dense(512, activation="relu", kernel_regularizer=l2(0.01))(input_data)
        encoded = Dropout(0.2, seed=self.random_state)(encoded)
        encoded = Dense(256, activation="relu", kernel_regularizer=l2(0.01))(encoded)
        encoded = Dropout(0.2, seed=self.random_state)(encoded)
        encoded = Dense(128, activation="relu", kernel_regularizer=l2(0.01))(encoded)
        encoded = Dense(self.encoding_dim, activation="relu")(encoded)
        # encoded = Dense(self.encoding_dim, activation="linear")(encoded)

        # Decoder
        decoded = Dense(128, activation="relu")(encoded)
        decoded = Dense(256, activation="relu")(decoded)
        decoded = Dense(512, activation="relu")(decoded)
        decoded = Dense(7, activation="sigmoid")(decoded)  # Output 7 features

        self.autoencoder_model = Model(input_data, decoded)
        self.encoder = Model(input_data, encoded)

        self.autoencoder_model.compile(optimizer=Adam(), loss="mean_squared_error")

        print("Autoencoder model built successfully.")

    def train_autoencoder(self, epochs=50, batch_size=256):
        """
        Trains the Autoencoder on the entire dataset.
        """
        all_data = np.vstack([data for data in self.scaled_ratios_data.values()])

        print("Starting model training...")
        self.autoencoder_model.fit(
            all_data,
            all_data,
            epochs=epochs,
            batch_size=batch_size,
            shuffle=True,
            validation_split=0.1  # 10% of data for validation
        )
        print("Model training completed.")

    def apply_autoencoder(self):
        """
        Applies autoencoder to each company's data to reduce dimensionality.
        """
        company_results = []

        for company, data in self.scaled_ratios_data.items():
            print(f"Processing company: {company}")
            company_encoded = []

            for time_point_data in data:
                time_point_data_reshaped = time_point_data.reshape(1, -1)  # (1, 7)
                if time_point_data_reshaped.shape[1] == 7:
                    encoded_time_point = self.encoder.predict(time_point_data_reshaped)
                    company_encoded.append(encoded_time_point.flatten())
                else:
                    print(f"Error: Invalid input shape {time_point_data_reshaped.shape}")

            company_results.append(np.array(company_encoded))

        self.reduced_data = np.array(company_results)
        return self.reduced_data

    def predict(self, data):
        """
        Reconstructs data using the trained autoencoder model.
        :param data: Input data to reconstruct.
        :return: Reconstructed data.
        """
        return self.autoencoder_model.predict(data)






