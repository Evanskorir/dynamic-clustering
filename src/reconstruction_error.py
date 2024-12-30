import numpy as np
import os
import pandas as pd


class ErrorReconstruction:
    def __init__(self, scaled_ratios_data, autoencoder_model):
        """
        Initialize ErrorReconstruction class.
        """
        self.scaled_ratios_data = scaled_ratios_data
        self.reconstructed_data = None
        self.autoencoder_model = autoencoder_model

    def calculate_reconstruction_error(self, metric: str = "mse"):
        """
        Calculates the reconstruction error (MSE) for each company using the autoencoder.
        """
        reconstruction_errors = {}
        self.reconstructed_data = {}

        for company, original_data in self.scaled_ratios_data.items():
            print(f"Calculating reconstruction error for company: {company}")

            reconstructed_data = self.autoencoder_model.predict(original_data)
            self.reconstructed_data[company] = reconstructed_data

            if metric == "mse":
                error = np.mean((original_data - reconstructed_data) ** 2, axis=1)
            elif metric == "mae":
                error = np.mean(np.abs(original_data - reconstructed_data), axis=1)
            else:
                raise ValueError(f"Unsupported metric: {metric}")

            reconstruction_errors[company] = error

        self.save_reconstructed_data()
        self.save_reconstruction_errors(reconstruction_errors)

        print("Reconstruction error calculation completed.")
        return self.reconstructed_data, reconstruction_errors

    def save_reconstructed_data(self, filename="reconstruction/reconstructed_data.csv"):
        """
        Saves the reconstructed data to a CSV file.
        """
        os.makedirs(os.path.dirname(filename), exist_ok=True)

        all_reconstructed_data = []
        for company, reconstructed_data in self.reconstructed_data.items():
            df = pd.DataFrame(reconstructed_data)
            df['Company'] = company
            all_reconstructed_data.append(df)

        final_df = pd.concat(all_reconstructed_data, ignore_index=True)
        final_df.to_csv(filename, index=False)
        print(f"Reconstructed data saved to {filename}")

    @staticmethod
    def save_reconstruction_errors(errors, filename="reconstruction/reconstruction_errors.csv"):
        """
        Saves the reconstruction errors to a CSV file.
        """
        os.makedirs(os.path.dirname(filename), exist_ok=True)

        all_errors = []
        for company, error in errors.items():
            error_df = pd.DataFrame(error, columns=['Reconstruction Error'])
            error_df['Company'] = company
            all_errors.append(error_df)

        final_error_df = pd.concat(all_errors, ignore_index=True)
        final_error_df.to_csv(filename, index=False)
        print(f"Reconstruction errors saved to {filename}")
