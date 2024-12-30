
from scipy.spatial.distance import cdist
from tslearn.metrics import cdist_dtw


class RatiosPairwiseDistance:
    def __init__(self, reduced_time_series_data):
        """
        :param reduced_time_series_data: Dictionary where keys are ratio names, and values
         are 2D arrays (companies x time points).
        """
        self.distance_matrix = None
        self.time_series_data = reduced_time_series_data  # Dictionary of {ratio: matrix}
        self.ratio_distance_matrices = {}  # Stores T x T matrices for each ratio

        # Compute pairwise distance for the entire time series (companies)
        self.compute_pairwise_distance(metric="dtw")

        # Compute pairwise distance for time points (for each ratio)
        self.compute_distance_matrices_for_all_ratios(metric="euclidean")

    def compute_pairwise_distance(self, metric="dtw"):
        """
        Compute pairwise distances using the specified metric.
        """
        if metric == "dtw":
            self.distance_matrix = cdist_dtw(self.time_series_data)
        else:
            raise ValueError("Invalid metric specified: Only 'dtw' is supported.")
        return self.distance_matrix

    def compute_time_point_distances(self, ratio_name, metric="euclidean"):
        """
        Compute pairwise distances between time points for a given ratio.
        :param ratio_name: The name of the ratio for which to compute distances.
        :param metric: Distance metric (default is Euclidean).
        :return: T x T distance matrix (T = number of time points).
        """
        if ratio_name not in self.time_series_data:
            raise ValueError(f"Ratio '{ratio_name}' not found in time series data.")

        # Extract the T x N matrix for the specified ratio (transpose to get time points as rows)
        data_matrix = self.time_series_data[ratio_name].T  # Shape (T, N)

        # Compute pairwise distances between time points
        distance_matrix = cdist(data_matrix, data_matrix, metric=metric)
        return distance_matrix

    def compute_distance_matrices_for_all_ratios(self, metric="euclidean"):
        """
        Compute T x T distance matrices for all ratios and store them in a dictionary.
        :param metric: Distance metric (default is Euclidean).
        :return: Dictionary of distance matrices for each ratio.
        """
        # List of all ratio names as provided
        ratio_names = [
            "Market Share",
            "Claims Paid Ratio",
            "Claims Incurred Ratio",
            "Underwriting Profits Ratio",
            "Expense Ratio",
            "Combined Ratio",
            "Claims Payout Ratio"
        ]

        # Compute distance matrices for each ratio and store in the dictionary
        for ratio_name in ratio_names:
            if ratio_name in self.time_series_data:
                self.ratio_distance_matrices[ratio_name] = self.compute_time_point_distances(
                    ratio_name, metric
                )
            else:
                print(f"Warning: Data for '{ratio_name}' not available.")

        return self.ratio_distance_matrices
