import numpy as np

from scipy.spatial.distance import cdist
from tslearn.metrics import cdist_dtw


class RatiosPairwiseDistance:
    def __init__(self, reduced_time_series_data, selected_range=None):
        """
        :param reduced_time_series_data:
            - If numpy array (shape: N x T x F): time series data across companies
            - If dict: {ratio_name: matrix} for ratio-based distance matrices
        :param selected_range: Tuple (start, end) to slice time series across time steps
        """
        self.distance_matrix = None
        self.ratio_distance_matrices = {}
        self.selected_range = selected_range

        if isinstance(reduced_time_series_data, dict):
            # Ratio-based case (dict of 2D matrices)
            self.time_series_data = self._slice_dict_data(reduced_time_series_data)
            self.compute_distance_matrices_for_all_ratios(metric="euclidean")
        elif isinstance(reduced_time_series_data, np.ndarray):
            # Time series case (N x T x F)
            self.time_series_data = self._slice_array_data(reduced_time_series_data)
            self.compute_pairwise_distance(metric="dtw")
        else:
            raise ValueError("Unsupported data format for reduced_time_series_data.")

    def _slice_array_data(self, data):
        if self.selected_range:
            start, end = self.selected_range
            return data[:, start:end + 1, :]
        return data

    def _slice_dict_data(self, data_dict):
        if not self.selected_range:
            return data_dict

        start, end = self.selected_range
        sliced_dict = {}
        for ratio_name, matrix in data_dict.items():
            if matrix.shape[1] <= end:
                print(f"Warning: Not enough time points in '{ratio_name}' for selected range.")
                continue
            sliced_dict[ratio_name] = matrix[:, start:end + 1]
        return sliced_dict

    def compute_pairwise_distance(self, metric="dtw"):
        """
        Compute pairwise DTW distances across companies.
        Shape: N x T x F → output: N x N
        """
        if metric == "dtw":
            self.distance_matrix = cdist_dtw(self.time_series_data)
        else:
            raise ValueError("Invalid metric specified: Only 'dtw' is supported.")
        return self.distance_matrix

    def compute_time_point_distances(self, ratio_name, metric="euclidean"):
        """
        Compute pairwise distances between time points for a given ratio.
        Input shape: N x T → Transposed to T x N for time-based comparison
        """
        if ratio_name not in self.time_series_data:
            raise ValueError(f"Ratio '{ratio_name}' not found in time series data.")

        data_matrix = self.time_series_data[ratio_name].T  # T x N
        distance_matrix = cdist(data_matrix, data_matrix, metric=metric)
        return distance_matrix

    def compute_distance_matrices_for_all_ratios(self, metric="euclidean"):
        """
        Compute T x T distance matrices for all ratios and store them.
        """
        ratio_names = [
            "Market Share",
            "Claims Paid Ratio",
            "Claims Incurred Ratio",
            "Underwriting Profits Ratio",
            "Expense Ratio",
            "Combined Ratio",
            "Claims Payout Ratio"
        ]

        for ratio_name in ratio_names:
            if ratio_name in self.time_series_data:
                self.ratio_distance_matrices[ratio_name] = self.compute_time_point_distances(
                    ratio_name, metric
                )
            else:
                print(f"Warning: Data for '{ratio_name}' not available.")

        return self.ratio_distance_matrices

