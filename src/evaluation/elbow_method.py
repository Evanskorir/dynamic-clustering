from tslearn.clustering import TimeSeriesKMeans


class ElbowMethod:
    def __init__(self, max_clusters=10):
        self.max_clusters = max_clusters

    def compute_inertia(self, input_data):
        distortions = []
        for k in range(1, self.max_clusters + 1):
            kmeans = TimeSeriesKMeans(n_clusters=k, metric="dtw", verbose=True)
            kmeans.fit(input_data)
            distortions.append(kmeans.inertia_)
        return distortions
