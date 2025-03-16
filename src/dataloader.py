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

    def __init__(self, include_reinsurers: bool = False):
        """
        Initializes the DataLoader with options to include reinsurers.

        Args:
            include_reinsurers (bool): Whether to include reinsurers in the loaded data.
        """
        self.include_reinsurers = include_reinsurers

        # Define unique file paths for each Excel file
        self.yearly_data_path = os.path.join(PROJECT_PATH, "../data", self.data_files["yearly"])
        self.quarterly_data_path = os.path.join(PROJECT_PATH, "../data", self.data_files["quarterly"])
        self.yearly_labels_path = os.path.join(PROJECT_PATH, "../data", self.data_files["yearly_labels"])
        self.quarterly_labels_path = os.path.join(PROJECT_PATH, "../data", self.data_files["quarterly_labels"])

        # Load data from both yearly and quarterly files
        self.yearly_medical_data = self._load_insurance_data(self.yearly_data_path)
        self.quarterly_medical_data = self._load_insurance_data(self.quarterly_data_path)

        # Load labels for yearly and quarterly data
        self.yearly_labels = self._load_labels(self.yearly_labels_path)
        self.quarterly_labels = self._load_labels(self.quarterly_labels_path)

    def _load_insurance_data(self, file_path):
        """
        Loads the insurance data from the specified file.

        Args:
            file_path (str): The path to the insurance data Excel file.

        Returns:
            dict: A dictionary where keys are sheet names and values are data as numpy arrays.
        """
        workbook = xlrd.open_workbook(file_path, on_demand=True)
        insurance_data = {}

        all_sheet_names = workbook.sheet_names()
        sheet_names_to_load = all_sheet_names if self.include_reinsurers else all_sheet_names[5:]

        for sheet_name in sheet_names_to_load:
            sheet = workbook.sheet_by_name(sheet_name)
            sheet_data = np.array([sheet.row_values(i) for i in range(sheet.nrows)])
            workbook.unload_sheet(sheet_name)
            insurance_data[sheet_name] = sheet_data

        return insurance_data

    def _load_labels(self, labels_path):
        """
        Loads labels from the specified labels file.

        Args:
            labels_path (str): The path to the labels Excel file.

        Returns:
            dict: A dictionary where keys are sheet names and values are label data as numpy arrays.
        """
        workbook = xlrd.open_workbook(labels_path, on_demand=True)
        label_data = {}

        for sheet_name in workbook.sheet_names():
            sheet = workbook.sheet_by_name(sheet_name)
            sheet_labels = np.array([sheet.cell_value(row, 0) for row in
                                     range(sheet.nrows)])
            workbook.unload_sheet(sheet_name)
            label_data[sheet_name] = sheet_labels

        return label_data
