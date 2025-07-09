from sklearn.metrics import silhouette_score
from tslearn.metrics import cdist_dtw
from sklearn.metrics.pairwise import euclidean_distances


class ClusteringEvaluation:
    def __init__(self, input_data, cluster_labels, distance_metric="dtw",
                 selected_range=None):
        """
        :param input_data: 3D array (N, T, F) for time series
        :param cluster_labels: Cluster assignments (1D array)
        :param distance_metric: "dtw" or "euclidean"
        :param selected_range: Tuple (start, end) to slice input_data along time axis
        """
        self.input_data = self._slice_data(input_data, selected_range)
        self.cluster_labels = cluster_labels
        self.distance_metric = distance_metric

    def _slice_data(self, data, selected_range):
        if selected_range:
            start, end = selected_range
            return data[:, start:end + 1, :]
        return data

    def compute_dtw_distance_matrix(self):
        return cdist_dtw(self.input_data)

    def compute_euclidean_distance_matrix(self):
        # Flatten 3D to 2D for Euclidean: shape (N, T*F)
        flat_data = self.input_data.reshape(self.input_data.shape[0], -1)
        return euclidean_distances(flat_data)

    def compute_silhouette_score(self):
        if self.distance_metric == "dtw":
            distance_matrix = self.compute_dtw_distance_matrix()
        elif self.distance_metric == "euclidean":
            distance_matrix = self.compute_euclidean_distance_matrix()
        else:
            raise ValueError(f"Unsupported distance metric: {self.distance_metric}")

        return silhouette_score(distance_matrix, self.cluster_labels,
                                metric="precomputed")

    def evaluate_clustering(self):
        score = self.compute_silhouette_score()
        print(f"Silhouette Score: {score:.4f}")
        return score
