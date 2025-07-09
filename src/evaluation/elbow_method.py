from tslearn.clustering import TimeSeriesKMeans


class ElbowMethod:
    def __init__(self, max_clusters=10, selected_range=None):
        self.max_clusters = max_clusters
        self.selected_range = selected_range

    def _slice_data(self, data):
        if self.selected_range:
            start, end = self.selected_range
            return data[:, start:end + 1, :]
        return data

    def compute_inertia(self, input_data):
        sliced_data = self._slice_data(input_data)
        distortions = []
        for k in range(1, self.max_clusters + 1):
            kmeans = TimeSeriesKMeans(n_clusters=k, metric="dtw", verbose=False)
            kmeans.fit(sliced_data)
            distortions.append(kmeans.inertia_)
        return distortions
