import streamlit as st
from src.controller import InsuranceAnalysisController
from src.dataloader import DataLoader
from src.ratios import InsuranceRatios


class DashboardController:
    def __init__(self, include_reinsurers, reduction_method, n_clusters):
        self.include_reinsurers = include_reinsurers
        self.reduction_method = reduction_method
        self.n_clusters = n_clusters

        self.data_loader = None
        self.ratios_data = None
        self.analysis = None

        self._run_pipeline()

    def _run_pipeline(self):
        # Step 1: Load Data
        self.data_loader = DataLoader(include_reinsurers=self.include_reinsurers)

        # Step 2: Compute Ratios
        self.ratios_data = InsuranceRatios(
            data=self.data_loader.quarterly_medical_data,
            include_reinsurers=self.include_reinsurers
        )

        # Step 3: Run Analysis Controller
        self.analysis = InsuranceAnalysisController(
            yearly_data=self.data_loader.yearly_medical_data,
            data=self.ratios_data,
            quarterly_labels=self.data_loader.quarterly_labels,
            yearly_labels=self.data_loader.yearly_labels,
            include_reinsurers=self.include_reinsurers,
            reduction_method=self.reduction_method
        )

        self.analysis.run_analysis(reduction_method=self.reduction_method,
                                   n_clusters=self.n_clusters)

    def get_cluster_labels(self):
        return self.analysis.clusters

    def get_companies(self):
        return self.analysis.companies

    def get_plotter(self):
        return self.analysis.plotter

    def get_distance_matrix(self):
        return self.analysis.distance_mtx

    def get_cluster_centers(self):
        return self.analysis.dtw_cluster.get_cluster_centers()

    def get_reduced_data(self):
        return self.analysis.reduced_time_series_data

    def get_reconstruction_errors(self):
        return self.analysis.reconstruction_errors

    def get_processed_yearly_data(self):
        return self.analysis.processed_yearly_data

    def get_yearly_labels(self):
        return self.analysis.yearly_labels

    def get_yearly_headers(self):
        return self.analysis.yearly_cols
