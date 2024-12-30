from sklearn.metrics import silhouette_score
from tslearn.metrics import cdist_dtw
from sklearn.metrics.pairwise import euclidean_distances


class ClusteringEvaluation:
    def __init__(self, input_data, cluster_labels, distance_metric="dtw"):
        """
        Initialize the ClusteringEvaluation class.
        :param input_data: The scaled time-series data or dim reduced data
        (3D numpy array or similar).
        :param cluster_labels: The cluster labels for each time series (1D numpy array).
        :param distance_metric: The distance metric to use ("dtw" or "euclidean").
        """
        self.input_data = input_data
        self.cluster_labels = cluster_labels
        self.distance_metric = distance_metric

    def compute_dtw_distance_matrix(self):
        """
        Compute the pairwise distance matrix using DTW.
        :return: Pairwise distance matrix (2D numpy array).
        """

        # Compute the pairwise DTW distance matrix
        return cdist_dtw(self.input_data)

    def compute_euclidean_distance_matrix(self):
        """
        Compute the pairwise Euclidean distance matrix.
        :return: Pairwise distance matrix (2D numpy array).
        """
        return euclidean_distances(self.input_data)

    def compute_silhouette_score(self):
        """
        Compute the silhouette score based on the clustering results.
        :return: Silhouette score (float).
        """
        # Compute the pairwise distance matrix
        if self.distance_metric == "dtw":
            distance_matrix = self.compute_dtw_distance_matrix()
        elif self.distance_metric == "euclidean":
            distance_matrix = self.compute_euclidean_distance_matrix()
        else:
            raise ValueError(f"Unsupported distance metric: {self.distance_metric}")

        # Compute silhouette score
        score = silhouette_score(distance_matrix, self.cluster_labels,
                                 metric="precomputed")
        return score

    def evaluate_clustering(self):
        """
        Evaluate clustering quality using the silhouette score.
        :return: Silhouette score (float).
        """
        silhouette = self.compute_silhouette_score()
        print(f"Silhouette Score: {silhouette:.4f}")
        return silhouette
