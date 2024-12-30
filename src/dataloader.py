import numpy as np
import os
import xlrd


PROJECT_PATH = os.path.dirname(os.path.realpath(__file__))


class DataLoader:
    """
    A class to load insurance data and labels from Excel files.
    """
    data_files = {
        "yearly": "Medical_yearly.xls",
        "quarterly": "Medical_quarterly.xls",
        "quarterly_labels": "quarterly_labels.xls",
        "yearly_labels": "yearly_labels.xls"
    }

    def __init__(self, data_type, include_reinsurers: bool = False):
        """
        Initializes the DataLoader with the specified data type and options.
        Args:
            data_type (str): The type of data to load, either 'yearly' or 'quarterly'.
            include_reinsurers (bool): Whether to include reinsurers in the loaded data.
        """
        if data_type not in ["yearly", "quarterly"]:
            raise ValueError(f"Unsupported data type: {data_type}")

        self.data_type = data_type
        self.include_reinsurers = include_reinsurers

        # Define file paths for all possible files
        self.yearly = os.path.join(PROJECT_PATH, "../data",
                                   self.data_files["yearly"])
        self.quarterly = os.path.join(PROJECT_PATH, "../data",
                                      self.data_files["quarterly"])
        self.quarterly_labels = os.path.join(PROJECT_PATH, "../data",
                                             self.data_files["quarterly_labels"])
        self.yearly_labels = os.path.join(PROJECT_PATH, "../data",
                                          self.data_files["yearly_labels"])

        # Select the active data and labels files based on the data type
        self._insurance_data_file = self.yearly if data_type == "yearly" else self.quarterly
        self.labels_file = self.yearly_labels if data_type == "yearly" else self.quarterly_labels

        # Load data and labels during initialization
        self.medical_data = self._load_insurance_data()
        self.labels = self._load_labels()

    def _load_insurance_data(self):
        """
        Loads the insurance data from the selected file.
        Returns:
        dict: A dictionary where keys are sheet names and values are data as numpy arrays.
        """
        workbook = xlrd.open_workbook(self._insurance_data_file, on_demand=True)
        data = {}

        all_sheet_names = workbook.sheet_names()
        sheet_names_to_load = all_sheet_names[5:] if not self.include_reinsurers else \
            all_sheet_names

        for sheet_name in sheet_names_to_load:
            sheet = workbook.sheet_by_name(sheet_name)
            sheet_data = np.array([sheet.row_values(i) for i in range(sheet.nrows)])
            workbook.unload_sheet(sheet_name)
            data[sheet_name] = sheet_data

        return data

    def _load_labels(self):
        """
        Loads the labels from the labels file for the selected insurance type.

        Returns:
            dict: A dictionary where keys are sheet names and values are label data
            as numpy arrays.
        """
        workbook = xlrd.open_workbook(self.labels_file, on_demand=True)
        labels = {}

        for sheet_name in workbook.sheet_names():
            sheet = workbook.sheet_by_name(sheet_name)
            sheet_labels = np.array([sheet.cell_value(row, 0) for row
                                     in range(sheet.nrows)])
            workbook.unload_sheet(sheet_name)
            labels[sheet_name] = sheet_labels

        return labels

