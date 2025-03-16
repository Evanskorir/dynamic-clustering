import numpy as np
import scipy.cluster.hierarchy as sch
from tslearn.clustering import TimeSeriesKMeans


class DTWClustering:
    def __init__(self, reduced_time_series_data, random_seed):
        """
        Initialize DTWClustering with preprocessed time-series data.
        :param reduced_time_series_data: Reduced-dimension time-series data.
        :param random_seed: Random seed for reproducibility.
        """

        self.time_series_data = reduced_time_series_data
        self.cluster_model = None
        self.cluster_labels = None
        self.random_state = random_seed
        self.linkage_matrix = None

    def perform_clustering(self, n_clusters=4):
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

    def perform_hierarchical_clustering(self, threshold=3.0,
                                        linkage="complete"):
        """
        Perform Hierarchical Clustering using the precomputed DTW distance matrix.
        :param threshold: Distance threshold for defining clusters.
        :param linkage: Linkage method ('single', 'complete', 'average', 'ward').
        """
        if self.time_series_data is None:
            raise ValueError("DTW distance matrix is not available. Ensure it's precomputed.")

        # Ensure the distance matrix is symmetric and float64
        distance_matrix = np.array(self.time_series_data, dtype=np.float64)

        self.linkage_matrix = sch.linkage(distance_matrix, method=linkage)

        # ✅ Extract cluster labels based on the threshold
        self.cluster_labels = sch.fcluster(self.linkage_matrix,
                                           t=threshold,
                                           criterion='distance')
