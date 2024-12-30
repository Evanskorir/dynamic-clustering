from tslearn.clustering import TimeSeriesKMeans


class DTWClustering:
    def __init__(self, reduced_time_series_data, random_seed):
        """
        Initialize DTWClustering with preprocessed time-series data.
        :param reduced_time_series_data: Reduced-dimension time-series data.
        :param random_seed: Random seed for reproducibility.
        """
        self.time_series_data = reduced_time_series_data
        # self.scaled_data = scaled_data
        # self.companies = companies
        self.cluster_model = None
        self.cluster_labels = None
        self.random_state = random_seed

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
