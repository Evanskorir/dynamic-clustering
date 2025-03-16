import numpy as np


class YearlyDataProcessor:
    def __init__(self, yearly_medical_data):
        self.yearly_medical_data = yearly_medical_data
        self.yearly_processed_data = None
        self.col_names = None
        self.process_yearly_data()

    @staticmethod
    def is_float(value):
        try:
            float(value)
            return True
        except ValueError:
            return False

    def process_yearly_data(self):
        processed_data = {}
        for company, data in self.yearly_medical_data.items():
            cleaned_data = []

            # Extract the column names (headers) from the first row
            if self.col_names is None:
                self.col_names = data[0]  # First row is the header

            # Iterate through the rows (skipping the header row)
            for row in data[1:]:  # Skipping the header row
                try:
                    # Convert the row into a numeric array and filter out invalid or non-numeric values
                    numeric_row = np.array([float(value) if self.is_float(value) else np.nan for value in row])

                    # Add cleaned numeric row to the list
                    cleaned_data.append(numeric_row)
                except ValueError:
                    # Skip rows with invalid data
                    continue

            # Convert the cleaned data into a numpy array
            cleaned_data = np.array(cleaned_data)

            # Ensure consistent number of years across all companies (padding if needed)
            num_years = cleaned_data.shape[1]
            max_years = max(data.shape[1] for data in self.yearly_medical_data.values())
            if num_years < max_years:
                padding = np.nan * (max_years - num_years)
                cleaned_data = np.column_stack([cleaned_data, padding])

            processed_data[company] = cleaned_data

        self.yearly_processed_data = processed_data
        return processed_data

