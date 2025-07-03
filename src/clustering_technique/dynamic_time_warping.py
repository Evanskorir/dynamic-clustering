import scipy.cluster.hierarchy as sch
from tslearn.clustering import TimeSeriesKMeans
from scipy.spatial.distance import pdist


class DTWClustering:
    def __init__(self, reduced_time_series_data, random_seed, approach: str = "pca"):
        """
        Initialize DTWClustering with preprocessed time-series data.
        :param reduced_time_series_data: Reduced-dimension time-series data.
        :param random_seed: Random seed for reproducibility.
        param approach: One of ["pca", "autoencoder", "lstm"]
        """

        self.time_series_data = reduced_time_series_data
        self.cluster_model = None
        self.cluster_labels = None
        self.random_state = random_seed
        self.linkage_matrix = None
        self.approach = approach.lower()

    def perform_clustering(self, n_clusters: int):
        """
        Perform KMeans clustering on the time-series data using DTW.
        :param n_clusters: Number of clusters.
        """
        self.cluster_model = TimeSeriesKMeans(
            n_clusters=n_clusters,
            n_init=2,
            metric="dtw",
            max_iter_barycenter=10,
            verbose=False,
            random_state=self.random_state,
        )

        # Fit the model and obtain cluster labels
        self.cluster_labels = self.cluster_model.fit_predict(self.time_series_data)

        print(f"Clustering completed using KMeans with {n_clusters} clusters!")

    def get_cluster_assignments(self):
        """
        Get the cluster assignments for each time-series.
        """
        if self.cluster_labels is None:
            raise ValueError("Clustering has not been performed yet.")
        return self.cluster_labels

    def get_cluster_centers(self):
        """
        Get the cluster centers after clustering.
        """
        if self.cluster_model is None or self.cluster_model.cluster_centers_ is None:
            raise ValueError("Clustering has not been performed yet.")
        return self.cluster_model.cluster_centers_

    def perform_hierarchical_clustering(self, threshold=4.0, linkage="ward"):
        """
        Perform Hierarchical Clustering using appropriate distance metric and input shape.
        :param threshold: Distance threshold for defining clusters.
        :param linkage: Linkage method.
        """
        if self.time_series_data is None:
            raise ValueError("Time-series data not available.")

        print(f"Performing hierarchical clustering with approach: {self.approach}")

        # Reshape if PCA or Autoencoder
        if self.approach in {"pca", "autoencoder"}:
            if self.time_series_data.ndim == 3:
                # Flatten last two dimensions: (samples, time * features)
                reshaped_data = self.time_series_data.reshape(
                    self.time_series_data.shape[0], -1
                )
            else:
                reshaped_data = self.time_series_data
            dist_vec = pdist(reshaped_data, metric="euclidean")

        elif self.approach == "lstm":
            # For LSTM, use DTW distance if needed (or fallback to Euclidean if appropriate)
            # Here we flatten to (samples, time * features) for simplicity
            reshaped_data = self.time_series_data.reshape(self.time_series_data.shape[0], -1)
            dist_vec = pdist(reshaped_data, metric="euclidean")

        else:
            raise ValueError(f"Unsupported approach for hierarchical "
                             f"clustering: {self.approach}")

        # Perform hierarchical clustering
        self.linkage_matrix = sch.linkage(dist_vec, method=linkage)
        self.cluster_labels = sch.fcluster(self.linkage_matrix, t=threshold,
                                           criterion="distance")

