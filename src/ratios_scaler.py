import numpy as np
import pandas as pd

from sklearn.preprocessing import MinMaxScaler, StandardScaler


class RatiosScaler:
    def __init__(self, ratios_data):
        self.scaled_data_list = None
        self.ratios_data = ratios_data
        self.scaled_ratios_data = {}

    @staticmethod
    def _clean_data(data):
        """
        Clean the data by converting to numeric and filling missing values with zeros.
        :param data: ndarray with raw data
        :return: ndarray with cleaned data
        """
        # Convert to DataFrame for easier handling
        df = pd.DataFrame(data)
        # Convert all columns to numeric, forcing errors to NaN
        df = df.apply(pd.to_numeric, errors='coerce')
        # Fill missing values with zeros
        df = df.fillna(0)
        # Convert back to ndarray
        return df.values

    def scale_the_data(self, scaler_type, method):
        """
        Scales the data using either within-company or global scaling.
        :param scaler_type: 'minmax' or 'standard'
        :param method: 'within' for scaling each company independently,
                       'global' for scaling all companies together.
        :return: Dictionary of scaled data and a 2D array of flattened data.
        """
        scaler = MinMaxScaler() if scaler_type.lower() == "minmax" else StandardScaler()

        if method == "within":
            # Scale each company's data independently
            self.scaled_ratios_data = {
                company: scaler.fit_transform(self._clean_data(data))
                for company, data in self.ratios_data.items()
            }
        elif method == "global":
            # Combine all companies' data for global scaling
            all_data = np.vstack([self._clean_data(data) for data in self.ratios_data.values()])
            scaler.fit(all_data)
            self.scaled_ratios_data = {
                company: scaler.transform(self._clean_data(data))
                for company, data in self.ratios_data.items()
            }
        else:
            raise ValueError("Invalid method. Choose 'within' or 'global'.")
        # Flatten the entire dataset for autoencoder training
        self.scaled_data_list = np.array(
            [scaled_data.flatten() for scaled_data in self.scaled_ratios_data.values()]
            )

        return self.scaled_ratios_data, self.scaled_data_list
