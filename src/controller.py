import numpy as np

from src.clustering_technique.dynamic_time_warping import DTWClustering
from src.dimension_reduction.autoencoder import Autoencoder
from src.dimension_reduction.pca import Pca
from src.dimension_reduction.lstm import LSTMFeatureFusion
from src.distance_matrix import RatiosPairwiseDistance
from src.plotter import Plotter
from src.ratios import InsuranceRatios
from src.ratios_scaler import RatiosScaler
from src.reconstruction_error import ErrorReconstruction


class InsuranceAnalysisController:
    def __init__(self, data: InsuranceRatios, labels, include_reinsurers, data_type: str):

        self.data_type = data_type
        self.data_loader = data
        self.labels = labels
        self.include_reinsurers = include_reinsurers
        self.time_series_data = None
        self.companies = None
        self.reduced_time_series_data = None

        self.reconstructed_data = None
        self.scaled_ratios_data = None
        self.distance_mtx = None
        self.autoencoder = None
        self.lstm_fusion = None
        self.reconstruction_errors = None

        self.clusters = None
        self.dtw_cluster = None
        self.plotter = None

    def load_data(self):
        max_len = max(len(series) for series in self.data_loader.ratios_data.values())
        self.time_series_data = np.array([
            np.pad(series, ((0, max_len - len(series)), (0, 0)), mode='constant',
                   constant_values=0)
            for series in self.data_loader.ratios_data.values()
        ])
        self.companies = list(self.data_loader.ratios_data.keys())

    def scale_the_data(self):
        print("Scaling the ratios data...")
        scaled_ratios_data = RatiosScaler(ratios_data=self.data_loader.ratios_data)
        scaled_ratios_data.scale_the_data(scaler_type="standard", method="within")
        self.scaled_ratios_data = scaled_ratios_data.scaled_ratios_data

    def apply_dimensionality_reduction(self, method: str = "autoencoder"):
        if method.lower() == "autoencoder":
            print("Applying Autoencoder for dimensionality reduction...")
            self.autoencoder = Autoencoder(
                scaled_ratios_data=self.scaled_ratios_data,
                encoding_dim=2,
                random_state=42
            )
            self.autoencoder.build_autoencoder()
            self.autoencoder.train_autoencoder(epochs=12, batch_size=16)
            self.reduced_time_series_data = self.autoencoder.apply_autoencoder()

        elif method.lower() == "pca":
            print("Applying PCA for dimensionality reduction...")
            pca = Pca(scaled_ratios_data=self.scaled_ratios_data)
            pca.PCA_apply(n_components=2)
            self.reduced_time_series_data = pca.reduced_data

        elif method.lower() == "lstm":
            print("Applying LSTM Feature Fusion for dimensionality reduction...")
            self.lstm_fusion = LSTMFeatureFusion(
                scaled_ratios_data=self.scaled_ratios_data,
                lstm_units=64,
                dropout_rate=0.2,
                random_state=42
            )
            self.lstm_fusion.build_lstm_model()
            self.lstm_fusion.train_lstm_model(epochs=12, batch_size=16)
            self.reduced_time_series_data = self.lstm_fusion.apply_lstm_model()
        else:
            raise ValueError(f"Unsupported dimensionality reduction method: {method}")

    def save_reconstruction_data(self):
        print("Calculating and saving reconstruction data...")
        error_reconstruction = ErrorReconstruction(
            scaled_ratios_data=self.scaled_ratios_data,
            autoencoder_model=self.autoencoder.autoencoder_model
        )

        self.reconstructed_data, self.reconstruction_errors = \
            error_reconstruction.calculate_reconstruction_error()
        print("Reconstruction data and errors saved successfully.")

    def get_pairwise_distance(self, option: str = "pairwise"):
        print("Calculating and plotting pairwise distance...")
        dist_mtx = RatiosPairwiseDistance(
            reduced_time_series_data=self.reduced_time_series_data)
        if option == "pairwise":
            self.distance_mtx = dist_mtx.distance_matrix
        else:
            self.distance_mtx = dist_mtx.ratio_distance_matrices

    def perform_clustering(self, n_clusters: int = 4):
        print(f"Performing KMeans clustering on reduced data...")
        dtw_cluster = DTWClustering(
            reduced_time_series_data=self.reduced_time_series_data,
            random_seed=42
        )
        dtw_cluster.perform_clustering(n_clusters=n_clusters)
        self.dtw_cluster = dtw_cluster
        self.clusters = dtw_cluster.get_cluster_assignments()

        cluster_groups = {}
        for idx, cluster in enumerate(self.clusters):
            cluster_groups.setdefault(cluster, []).append(self.companies[idx])

        for cluster, members in cluster_groups.items():
            print(f"Cluster {cluster + 1}: {members}")

    def get_evaluation_plot(self, evaluation_criteria):
        # Map evaluation criteria to the corresponding plotter methods
        evaluation_methods = {
            "silhouette": lambda: self.plotter.plot_silhouette_curve(
                input_data=self.reduced_time_series_data,
                max_clusters=12,
                metric="dtw"
            ),
            "elbow": lambda: self.plotter.plot_elbow(
                input_data=self.reduced_time_series_data,
                max_clusters=12)
        }

        plot_method = evaluation_methods.get(evaluation_criteria)
        if plot_method:
            plot_method()

    def initialize_plotter(self):
        print("Initializing Plotter...")
        cluster_centers = self.dtw_cluster.get_cluster_centers()
        self.plotter = Plotter(
            cluster_centers=cluster_centers,
            time_series_data=self.data_loader.ratios_data,
            data_scaled=self.scaled_ratios_data,
            cluster_labels=self.clusters,
            companies=self.companies,
            dtw_clustering=self.dtw_cluster,
            reduced_data=self.reduced_time_series_data,
            reconstructed_data=self.reconstructed_data,
            labels=self.labels,
            include_reinsurers=self.include_reinsurers
        )
        self.plotter.plot_distance_matrix(distance_matrix=self.distance_mtx)
        self.plotter.plot_cluster_scatter()
        self.plotter.plot_reconstruction_error(
            reconstruction_errors=self.reconstruction_errors)

    def plot_variables(self):
        # Step 4: Plot variables (Gross Premium Income, Claims Paid,
        # Claims Incurred, Underwriting Profits etc.)
        print("Plotting variables...")
        self.plotter.plot_time_series_heatmap(self.reduced_time_series_data)
        self.plotter.plot_variable_split(0, "Market Share")
        self.plotter.plot_variable_split(1, "Claims Paid Ratio")
        self.plotter.plot_variable_split(2, "Claims Incurred Ratio")
        self.plotter.plot_variable_split(3, "Underwriting Profits Ratio")
        self.plotter.plot_variable_split(4, "Expense Ratio")
        self.plotter.plot_variable_split(5, "Combined Ratio")
        self.plotter.plot_variable_split(6, "Claims Payout Ratio")

    def run_analysis(self, reduction_method):
        # Step 1: Load and Scale Data
        self.load_data()
        self.scale_the_data()

        # Step 2: Apply Dimensionality Reduction
        self.apply_dimensionality_reduction(method=reduction_method)

        # Step 3: Compute Pairwise Distance
        self.get_pairwise_distance()

        # Step 4: Perform Clustering
        self.perform_clustering()

        # Step 5: Initialize Plotter
        self.initialize_plotter()
        # Visualize the Variables (Financial ratios)
        self.plot_variables()

        # Step 6: Get Evaluation Plot
        self.get_evaluation_plot(evaluation_criteria="elbow")

        # Step 7 & 8: Plot 2D Dimensionality Reduction and Autoencoder-Specific Actions
        if reduction_method in ["autoencoder", "pca"]:
            self.plotter.plot_2d_dimension_reduction(company_names=self.companies,
                                                     method=reduction_method)

            if reduction_method == "autoencoder":
                self.save_reconstruction_data()
                self.plotter.plot_reconstruction_error(
                    reconstruction_errors=self.reconstruction_errors)





