import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, TimeDistributed, Dense, Dropout
from tensorflow.keras.optimizers import Adam


class LSTMFeatureFusion:
    def __init__(self, scaled_ratios_data, lstm_units=64, dropout_rate=0.2,
                 random_state=42):
        """
        Initializes the LSTM Feature Fusion class.
        :param scaled_ratios_data: Input data as a dictionary of DataFrames or arrays.
        :param lstm_units: Number of units in the LSTM layer.
        :param dropout_rate: Dropout rate for regularization.
        :param random_state: Seed for reproducibility.
        """
        self.scaled_ratios_data = scaled_ratios_data
        self.lstm_units = lstm_units
        self.dropout_rate = dropout_rate
        self.random_state = random_state

        self.model = None
        self.reduced_data = None

    def build_lstm_model(self):
        """
        Builds the LSTM model to reduce 7 features to 1 over time with explicit activations.
        """
        tf.random.set_seed(self.random_state)

        # Input shape: (time_steps, features) -> (41, 7)
        input_layer = Input(shape=(41, 7))

        # LSTM Layer with explicit activations
        lstm_out = LSTM(
            self.lstm_units,
            activation='tanh',  # Input and output activation
            recurrent_activation='sigmoid',  # Gate activation
            return_sequences=True
        )(input_layer)
        lstm_out = Dropout(self.dropout_rate)(lstm_out)

        # TimeDistributed Dense layer to reduce features to 1
        output_layer = TimeDistributed(Dense(1, activation='tanh'))(lstm_out)

        # Build and compile the model
        self.model = Model(inputs=input_layer, outputs=output_layer)
        self.model.compile(optimizer=Adam(), loss='mean_squared_error')

        print("LSTM model built successfully with explicit activations.")

    def train_lstm_model(self, epochs=50, batch_size=8):
        """
        Trains the LSTM model on the entire dataset.
        """
        # Combine all company data into a single array
        all_data = np.array([data for data in self.scaled_ratios_data.values()])

        print("Starting model training...")
        self.model.fit(
            all_data,
            all_data,  # Self-supervised: learning to reconstruct
            epochs=epochs,
            batch_size=batch_size,
            shuffle=True,
            validation_split=0.1
        )
        print("Model training completed.")

    def apply_lstm_model(self):
        """
        Applies the trained LSTM model to reduce dimensionality.
        """
        company_results = []

        for company, data in self.scaled_ratios_data.items():
            print(f"Processing company: {company}")
            reduced_series = self.model.predict(data[np.newaxis, :, :])  # Shape: (1, 41, 1)
            company_results.append(reduced_series.squeeze())

        self.reduced_data = np.array(company_results)  # Shape: (28, 41, 1)
        return self.reduced_data

    def predict(self, data):
        """
        Predicts reduced output for new data.
        :param data: Input data to transform.
        :return: Reduced dimensionality data.
        """
        return self.model.predict(data)
