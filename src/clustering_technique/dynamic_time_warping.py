import scipy.cluster.hierarchy as sch

from tslearn.clustering import TimeSeriesKMeans
from scipy.spatial.distance import pdist


class DTWClustering:
    def __init__(self, reduced_time_series_data, random_seed, approach: str = "pca"):
        """
        :param reduced_time_series_data: Time-series data (N x T x F)
        :param random_seed: Random seed for reproducibility
        :param approach: "pca", "autoencoder", or "lstm"
        """
        self.original_time_series_data = reduced_time_series_data
        self.time_series_data = reduced_time_series_data  # May be sliced later
        self.cluster_model = None
        self.cluster_labels = None
        self.random_state = random_seed
        self.linkage_matrix = None
        self.approach = approach.lower()

    def _slice_data(self, selected_range):
        if selected_range:
            start, end = selected_range
            return self.original_time_series_data[:, start:end+1, :]
        return self.original_time_series_data

    def perform_clustering(self, n_clusters: int, selected_range=None):
        """
        Perform DTW KMeans clustering
        :param n_clusters: Number of clusters
        :param selected_range: Tuple (start, end) for slicing time dimension
        """
        self.time_series_data = self._slice_data(selected_range)

        self.cluster_model = TimeSeriesKMeans(
            n_clusters=n_clusters,
            n_init=2,
            metric="dtw",
            max_iter_barycenter=10,
            verbose=False,
            random_state=self.random_state,
        )
        self.cluster_labels = self.cluster_model.fit_predict(self.time_series_data)
        print(f"Clustering completed using KMeans with {n_clusters} clusters!")

    def get_cluster_assignments(self):
        if self.cluster_labels is None:
            raise ValueError("Clustering has not been performed yet.")
        return self.cluster_labels

    def get_cluster_centers(self):
        if self.cluster_model is None or self.cluster_model.cluster_centers_ is None:
            raise ValueError("Clustering has not been performed yet.")
        return self.cluster_model.cluster_centers_

    def perform_hierarchical_clustering(self, threshold=4.0, linkage="ward",
                                        selected_range=None):
        """
        Perform hierarchical clustering using `pdist` on sliced time series data.
        :param threshold: Distance threshold to cut the dendrogram.
        :param linkage: Linkage method for hierarchical clustering.
        :param selected_range: Tuple (start, end) to slice time.
        """
        self.time_series_data = self._slice_data(selected_range)

        if self.time_series_data is None:
            raise ValueError("Time-series data not available.")

        print(f"Performing hierarchical clustering with approach: {self.approach}")

        if self.time_series_data.ndim == 3:
            reshaped_data = self.time_series_data.reshape(
                self.time_series_data.shape[0], -1)
        else:
            reshaped_data = self.time_series_data

        # For all approaches, use Euclidean distance on flattened data
        dist_vec = pdist(reshaped_data, metric="euclidean")

        self.linkage_matrix = sch.linkage(dist_vec, method=linkage)
        self.cluster_labels = sch.fcluster(self.linkage_matrix,
                                           t=threshold, criterion="distance")
