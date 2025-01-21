import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import os
import scipy.cluster.hierarchy as sch
import seaborn as sns

from matplotlib.colors import ListedColormap, LinearSegmentedColormap
from src.evaluation.silhouette import ClusteringEvaluation
from src.evaluation.elbow_method import ElbowMethod
from scipy.interpolate import make_interp_spline

matplotlib.use('agg')


class Plotter:
    def __init__(self, cluster_centers, time_series_data, data_scaled,
                 cluster_labels, companies, dtw_clustering, labels,
                 include_reinsurers, reduced_data=None, reconstructed_data=None):

        self.cluster_centers = cluster_centers
        self.time_series_data = time_series_data
        self.data_scaled = data_scaled
        self.cluster_labels = cluster_labels
        self.companies = companies
        self.dtw_clustering = dtw_clustering
        self.labels = labels
        self.reduced_data = reduced_data
        self.reconstructed_data = reconstructed_data
        self.include_reinsurers = include_reinsurers

    @staticmethod
    def _create_output_dir(output_subdir=""):
        """
        Creates a flexible output directory structure.
        """
        base_dir = "./plots/"
        if output_subdir:
            output_dir = os.path.join(base_dir, output_subdir)
        else:
            output_dir = base_dir
        os.makedirs(output_dir, exist_ok=True)
        return output_dir

    def _save_plot(self, filename, output_subdir=""):
        """
        Save the plot to the specified subdirectory.
        """
        output_dir = self._create_output_dir(output_subdir)
        filepath = os.path.join(output_dir, filename)
        plt.savefig(filepath, dpi=300, bbox_inches="tight")
        print(f"Plot saved to {filepath}")
        plt.close()

    def plot_distance_matrix(self, distance_matrix):
        """
        Plots the distance matrix computed using DTW distance, and orders it
        based on hierarchical clustering using the complete linkage method.
        Saves both the unordered and ordered heatmaps.
        """
        if distance_matrix is None:
            raise ValueError("Distance matrix is not available. "
                             "Ensure DTW clustering has been performed.")

        # Perform hierarchical clustering and reorder matrix
        ordered_distance_matrix, dendro_idx = self._reorder_distance_matrix(distance_matrix)

        # Plot and save unordered and ordered heatmaps
        self._plot_heatmap(distance_matrix, "Unordered DTW Distance Matrix",
                           "unordered_dtw_distance_matrix.png",
                           output_subdir="distance_matrix", add_colorbar=False)

        self._plot_heatmap(
            ordered_distance_matrix, "Ordered DTW Distance Matrix",
            "ordered_dtw_distance_matrix.png", output_subdir="distance_matrix",
            reordered_labels=dendro_idx, add_colorbar=True
        )

    def _plot_heatmap(self, matrix, title, filename, reordered_labels=None,
                      output_subdir="", add_colorbar=False):
        """
        Plots a heatmap for the given matrix, applies consistent styling,
        and saves the plot to a file.
        """
        fig, ax = plt.subplots(figsize=(16, 14))
        labels = np.array(self.companies) if reordered_labels is None else \
            np.array(self.companies)[reordered_labels]

        # Ensure no default colorbar is added (set cbar=False)
        cax = sns.heatmap(
            matrix,
            cmap="viridis",  # Consistent colormap
            annot=False,
            fmt=".1f",
            annot_kws={"size": 9, "weight": "bold", "color": "black"},
            xticklabels=labels,
            yticklabels=labels,
            linewidths=0,  # Remove grid lines
            linecolor='none',
            # linecolor='gray',  # Set line color to gray for clarity if necessary
            square=False,  # Ensure it's not square to retain aspect ratio
            cbar=False  # No default colorbar
        )

        self._style_plot(ax, title)

        # Add custom colorbar if the flag is set to True (for ordered matrix only)
        if add_colorbar:
            self._add_colorbar(fig, cax)

        plt.tight_layout()
        self._save_plot(filename, output_subdir)
        plt.close()

    @staticmethod
    def _reorder_distance_matrix(distance_matrix):
        """
        Reorders the distance matrix using hierarchical clustering with the
        complete linkage method.
        """
        linkage_matrix = sch.linkage(distance_matrix, method='complete')
        dendro_idx = sch.leaves_list(linkage_matrix)
        ordered_distance_matrix = distance_matrix[dendro_idx, :][:, dendro_idx]
        return ordered_distance_matrix, dendro_idx

    def _style_plot(self, ax, title):
        """
        Styles the heatmap plot with balanced bold and rotated labels.
        """
        # Abbreviate company names
        abbreviated_labels = [name if len(name) <= 15 else name[:12] + "..." for
                              name in self.companies]

        num_labels = len(abbreviated_labels)

        # Center the ticks by offsetting by 0.5
        ax.set_xticks(np.arange(num_labels) + 0.5)
        ax.set_yticks(np.arange(num_labels) + 0.5)
        ax.set_xticklabels(abbreviated_labels, fontsize=22, rotation=90, ha='right')
        ax.set_yticklabels(abbreviated_labels, fontsize=22, ha="right")

        # Customize tick size and thickness
        ax.tick_params(axis='x', which='both', labelsize=22, width=2.2, length=8)
        ax.tick_params(axis='y', which='both', labelsize=22, width=2.2, length=8)

        # Add grid lines for readability
        # ax.grid(visible=True, linestyle='--', linewidth=0.5, alpha=0.3)

        # Plot boundaries
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(2)

        plt.tight_layout(rect=[0, 0.05, 1, 1])

    @staticmethod
    def _style_plot2(ax, title):
        """
        Styles the heatmap plot with titles, labels, and tick parameters.
        """

        # Set fonts for tick labels
        plt.xticks(fontsize=22, rotation=90, fontweight='bold')
        plt.yticks(fontsize=22, rotation=0, fontweight='bold')

        ax.tick_params(axis='x', which='both', labelsize=22, width=2.2, length=8)
        ax.tick_params(axis='y', which='both', labelsize=22, width=2.2, length=8)

        # Customize the plot boundaries
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(2)

    @staticmethod
    def _add_colorbar(fig, cax):
        """
        Adds a custom colorbar to the heatmap plot.
        """
        cbar_ax = fig.add_axes((1.05, 0.2, 0.03, 0.79))  # (left, bottom, width, height)
        cbar = fig.colorbar(cax.collections[0], cax=cbar_ax)

        cbar.ax.tick_params(labelsize=20, colors="darkgreen", width=2)
        cbar.outline.set_visible(True)
        cbar.outline.set_linewidth(1.5)

        for tick in cbar.ax.get_yticklabels():
            tick.set_fontsize(25)
            tick.set_color("darkgreen")

        cbar.set_label('DTW Distance', fontsize=20, fontweight='bold',
                       color='darkgreen', labelpad=20)

    def plot_time_series_heatmap(self, time_series):
        """
        Plots a heatmap of the reduced time series data for each company.
        """

        if time_series is None:
            raise ValueError("Reduced time series data is not available.")

        # Ensure the input is a NumPy array for consistency
        heatmap_array = np.array(time_series)

        # Calculate the aspect ratio to maintain a rectangular shape
        aspect_ratio = heatmap_array.shape[1] / heatmap_array.shape[0]
        fig, ax = plt.subplots(figsize=(30, 14))

        # Plot the heatmap using 'viridis' colormap for smooth gradients
        cax = sns.heatmap(
            heatmap_array,
            cmap="viridis",
            xticklabels=False,
            yticklabels=False,
            cbar=False,  # Disable default colorbar for custom styling
            linewidths=0,
            ax=ax,
            linecolor='none'
        )

        # Add a custom colorbar
        cbar = fig.colorbar(
            cax.collections[0],
            ax=ax,
            orientation='vertical',
            fraction=0.03,
            pad=0.04
        )
        cbar.ax.set_ylabel("LSTM reduced data", fontsize=20, fontweight='bold',
                           color="darkgreen", labelpad=20)
        cbar.ax.tick_params(labelsize=20, colors="darkgreen")
        cbar.outline.set_linewidth(1.5)

        # Set x-axis ticks for time points
        num_time_points = heatmap_array.shape[1]
        quarterly_labels = self.labels.get('Sheet1', [])
        ax.set_xticks(np.arange(num_time_points) + 0.5)
        ax.set_xticklabels(quarterly_labels, rotation=90,
                           fontsize=18, fontweight='bold', ha='center')

        # Set y-axis ticks for companies
        ax.set_yticks(np.arange(len(self.companies)) + 0.5)
        ax.set_yticklabels(self.companies, fontsize=18, fontweight='bold', va='center')

        # Customize tick size and thickness
        ax.tick_params(axis='x', which='both', labelsize=18, width=2, length=8)
        ax.tick_params(axis='y', which='both', labelsize=18, width=2, length=8)

        # Enhance plot boundaries
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(2)

        plt.tight_layout(rect=[0, 0, 0.9, 1])

        # Save the heatmap
        filename = "time_series_heatmap.png"
        self._save_plot(filename, output_subdir="time_series")
        plt.close()
        print(f"Time series heatmap saved to '{filename}'")

    def plot_variable_split(self, variable_index, variable_name):
        """
        Plots a heatmap for the variable across all companies.
        Ensures a rectangular shape and centered tick labels.
        """
        # Prepare the data in dictionary format for all variables
        variable_data = {
            company: values[:, variable_index].astype(float)
            for company, values in self.time_series_data.items()
            if values.ndim == 2 and values.shape[1] > variable_index
        }

        # Determine companies to include
        if self.include_reinsurers:
            companies = list(variable_data.keys())[5:]  # Exclude the first 5 companies
        else:
            companies = list(variable_data.keys())  # Include all companies

        # Gather data into a 2D array
        heatmap_data = [variable_data[company] for company in companies]
        heatmap_array = np.array(heatmap_data)  # Directly convert to a numpy array

        # Adjust aspect ratio for a more rectangular plot
        aspect_ratio = len(heatmap_array[0]) / len(companies)  # Based on data dimensions
        fig, ax = plt.subplots(figsize=(30, 14))  # Explicitly set a rectangular size

        # Plot the heatmap without a default colorbar
        cax = sns.heatmap(
            heatmap_array,
            cmap="viridis",
            xticklabels=False,
            yticklabels=False,
            cbar=False,  # Disable default colorbar
            linewidths=0,  # Disable grid lines
            ax=ax,
            linecolor='none',
        )

        # Add a single custom colorbar at the far right
        cbar = fig.colorbar(
            cax.collections[0],
            ax=ax,
            orientation='vertical',
            fraction=0.03,  # Adjust colorbar width
            pad=0.04,
            format='%d%%'
        )
        cbar.ax.set_ylabel(variable_name, fontsize=20, fontweight='bold',
                           color="darkgreen", labelpad=20)
        cbar.ax.tick_params(labelsize=20, colors="darkgreen")
        cbar.outline.set_linewidth(1.5)

        # Adjust x-axis labels to match quarterly time points and center them
        num_time_points = heatmap_array.shape[1]
        quarterly_labels = self.labels.get('Sheet1', [])
        ax.set_xticks(np.arange(num_time_points) + 0.5)  # Center ticks
        ax.set_xticklabels(quarterly_labels, rotation=90,
                           fontsize=18, fontweight='bold', ha='center')

        # Set y-axis ticks for companies and center them
        ax.set_yticks(np.arange(len(companies)) + 0.5)
        ax.set_yticklabels(companies, fontsize=18, fontweight='bold', va='center')

        # Customize the tick size and thickness for both axes
        ax.tick_params(axis='x', which='both', labelsize=18, width=2, length=8)
        ax.tick_params(axis='y', which='both', labelsize=18, width=2, length=8)

        # Customize the plot boundaries
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(2)  # Strong boundary lines

        plt.tight_layout(rect=[0, 0, 0.9, 1])

        # Save the plot
        filename = f"heatmap_{variable_name.replace(' ', '_').lower()}.pdf"
        self._save_plot(filename, output_subdir=f"heatmaps/{variable_name}")
        plt.close()
        print(f"Heatmap for '{variable_name}' saved to '{filename}'")

    def plot_elbow(self, input_data, max_clusters=10):
        elbow = ElbowMethod(max_clusters)
        distortions = elbow.compute_inertia(input_data)

        # Visualization setup
        plt.figure(figsize=(14, 8), dpi=300)  # High DPI for consistent font rendering

        # Plot smooth, continuous curve with markers
        plt.plot(range(1, max_clusters + 1), distortions, color='dodgerblue',
                 linewidth=2, marker='o', markersize=8,
                 markerfacecolor='white', markeredgewidth=2, linestyle='-', alpha=0.9)  # Unfilled markers

        # Title and labels with modern, readable fonts
        plt.xlabel("Number of Clusters", fontsize=20, fontweight='bold', labelpad=15)
        plt.ylabel("Inertia (Sum of Squared Distances)", fontsize=20,
                   fontweight='bold', labelpad=15)

        # Refine ticks for readability
        plt.xticks(range(1, max_clusters + 1), fontsize=18, fontweight='bold')
        plt.yticks(fontsize=18, fontweight='bold')

        # Add grid lines only along the y-axis, for clarity without clutter
        plt.grid(axis='y', linestyle='--', alpha=0.4)

        # Apply a light, elegant background color to the plot
        plt.gca().set_facecolor('#f5f5f5')

        # Add a horizontal line at y=0 for reference
        plt.axhline(y=0, color='black', linewidth=1.2, linestyle='--')

        # Customize strong boundaries for the plot
        ax = plt.gca()
        ax.spines['top'].set_linewidth(1.5)  # Top boundary
        ax.spines['right'].set_linewidth(1.5)  # Right boundary
        ax.spines['left'].set_linewidth(1.5)  # Left boundary
        ax.spines['bottom'].set_linewidth(1.5)  # Bottom boundary

        # Explicitly set the y-axis range to ensure a clean look
        plt.ylim(bottom=0, top=np.max(distortions) + 0.1 * np.max(distortions))

        # Tight layout to avoid clipping of elements
        plt.tight_layout()

        # Save and display the plot
        self._save_plot("elbow_method.png", output_subdir="evaluation")

    def plot_silhouette_curve(self, input_data, max_clusters=12, metric="dtw"):
        silhouette_scores = []
        clusters_range = range(2, max_clusters + 1)

        # Compute silhouette scores for each number of clusters
        for n_clusters in clusters_range:
            self.dtw_clustering.perform_clustering(n_clusters=n_clusters)
            cluster_labels = self.dtw_clustering.get_cluster_assignments()
            evaluation = ClusteringEvaluation(input_data=input_data,
                                              cluster_labels=cluster_labels,
                                              distance_metric=metric)
            silhouette_scores.append(evaluation.evaluate_clustering())

        # Visualization setup
        plt.figure(figsize=(14, 8), dpi=300)  # Use higher DPI to ensure font consistency

        # Normalize silhouette scores to [0, 1] range for color mapping
        normed_scores = (np.array(silhouette_scores) - np.nanmin(silhouette_scores)) / (
                np.nanmax(silhouette_scores) - np.nanmin(silhouette_scores))

        # Plot smooth, continuous curve with markers
        plt.plot(clusters_range, silhouette_scores, color='dodgerblue',
                 linewidth=2, marker='o', markersize=8,
                 markerfacecolor='white', markeredgewidth=2, linestyle='-', alpha=0.9)

        max_score = np.max(silhouette_scores)
        buffer_value = 0.1
        plt.ylim(0, max_score + buffer_value)

        # Title and labels with modern, readable fonts
        plt.xlabel("Number of Clusters", fontsize=18, fontweight='bold', labelpad=15)
        plt.ylabel("Silhouette Score", fontsize=18, fontweight='bold', labelpad=15)

        # Refine ticks for readability
        plt.xticks(clusters_range, fontsize=18, fontweight='bold')
        plt.yticks(fontsize=18, fontweight='bold')

        # Add grid lines only along the y-axis, for clarity without clutter
        plt.grid(axis='y', linestyle='--', alpha=0.4)

        plt.gca().set_facecolor('#f5f5f5')
        plt.axhline(y=0, color='black', linewidth=1.2, linestyle='--')

        # Display plot and save with a high-quality resolution
        self._save_plot("silhouette.png", output_subdir="evaluation")

    def plot_2d_dimension_reduction(self, company_names, method: str):
        """
        Visualizes the reduced 2D data from Autoencoder and saves the plot.
        Adds annotations for each company, cluster labeling, and enhanced visuals.

        Args:
        company_names (list): A list of company names or identifiers (length should match the number of points).
        """
        if self.reduced_data is None:
            raise ValueError(f"Reduced data (from {method}) is not available.")

        num_points = self.reduced_data.shape[0]  # Number of companies

        # Ensure the correct number of points in reduced data and cluster labels
        if len(self.cluster_labels) != num_points:
            raise ValueError(
                f"Mismatch between the number of reduced data points "
                f"({num_points}) and cluster labels ({len(self.cluster_labels)}).")

        # Average over the years for each company to get a single 2D point per company
        x_vals = np.mean(self.reduced_data[:, :, 0], axis=1)  # Average first reduced dimension over time
        y_vals = np.mean(self.reduced_data[:, :, 1], axis=1)  # Average second reduced dimension over time

        # Create the scatter plot of the reduced 2D data
        plt.figure(figsize=(16, 14))
        # Adjust the cluster labels to start from 1
        cluster_labels_adjusted = self.cluster_labels + 1
        num_clusters = len(np.unique(cluster_labels_adjusted))

        # Create a ListedColormap for the clusters
        colors = ["darkgreen", "purple", 'red', 'cyan', 'darkgray', 'indigo']
        custom_cmap = ListedColormap(colors[:num_clusters])

        # Plot the points with the cluster labels as colors
        scatter = plt.scatter(x_vals, y_vals, c=cluster_labels_adjusted,
                              cmap=custom_cmap,
                              edgecolor='k', s=200, alpha=0.8, marker='o')

        plt.xlabel("Reduced Dimension 1", fontsize=20, fontweight='bold',
                   family='Arial', labelpad=15)
        plt.ylabel("Reduced Dimension 2", fontsize=20, fontweight='bold',
                   family='Arial', labelpad=15)

        # Customize ticks and tick labels
        plt.xticks(fontsize=16, fontweight='bold', family='Arial', color='gray')
        plt.yticks(fontsize=16, fontweight='bold', family='Arial', color='gray')
        plt.tick_params(axis='both', which='major', width=2, length=10, color='black')

        # Add annotations for each point (company), using company names
        for i, (x, y) in enumerate(zip(x_vals, y_vals)):
            plt.annotate(
                company_names[i],  # Display company name
                (x, y),
                textcoords="offset points",
                xytext=(0, 15),
                ha='center',
                fontsize=12,
                color='black',
                fontweight='bold',
                bbox=dict(facecolor='white', alpha=0.85, edgecolor='black',
                          boxstyle='round,pad=0.4', linewidth=0.5),
                zorder=5,
            )

        # Add a legend for the clusters with bold title and larger font size
        handles = [plt.Line2D([0], [0], marker='o', color='w',
                              markerfacecolor=colors[i], markersize=12)
                   for i in range(num_clusters)]

        plt.legend(handles, [f"Cluster {i}" for i in range(1, num_clusters + 1)],
                   title="Clusters", loc="upper right", fontsize=14,
                   title_fontsize=16, prop={'weight': 'bold'}, shadow=True)

        # Add grid and customize boundary
        plt.grid(color='gray', linestyle='--', linewidth=0.5, alpha=0.6)
        for spine in plt.gca().spines.values():
            spine.set_linewidth(1.5)
            spine.set_color('black')

        # Set a light background color for improved layout
        plt.gca().set_facecolor('#f2f2f2')

        # Save the plot to the plots directory
        self._save_plot(f"{method}_reduced_2d_plot.pdf", output_subdir="projection")

    def plot_reconstruction_error(self, reconstruction_errors):
        """
        Plots a heatmap of reconstruction errors for all companies with a rectangular shape,
        centered ticks, a single customized color bar, no grid lines, and strong boundary lines.
        """
        if not reconstruction_errors:
            print("No valid reconstruction errors calculated.")
            return

        # Prepare reconstruction error data in matrix format
        company_names = list(reconstruction_errors.keys())
        max_time_points = max(len(errors) for errors in reconstruction_errors.values())
        reconstruction_matrix = []

        # Build reconstruction matrix with actual data
        for company in company_names:
            errors = reconstruction_errors[company]
            # Pad shorter arrays with NaNs to align all companies' data
            padded_errors = np.pad(errors, (0, max_time_points - len(errors)),
                                   constant_values=np.nan)
            reconstruction_matrix.append(padded_errors)

        # Convert to numpy array for heatmap
        heatmap = np.array(reconstruction_matrix)

        # Adjust aspect ratio for a rectangular plot
        aspect_ratio = len(heatmap[0]) / len(company_names)
        # fig, ax = plt.subplots(figsize=(20 * aspect_ratio, 10))
        fig, ax = plt.subplots(figsize=(30, 14))

        # Plot the heatmap without a default colorbar
        cax = sns.heatmap(
            heatmap,
            cmap="viridis",
            xticklabels=False,
            yticklabels=False,
            cbar=False,  # Disable default colorbar
            linewidths=0,  # Remove grid lines
            ax=ax,
        )

        # Add a single custom colorbar with proper spacing
        cbar = fig.colorbar(
            cax.collections[0],
            ax=ax,
            orientation='vertical',
            fraction=0.03,
            pad=0.04,
        )
        cbar.ax.set_ylabel('Reconstruction Error', fontsize=22, fontweight='bold',
                           color="darkgreen", labelpad=20)
        cbar.ax.tick_params(labelsize=22, colors="darkgreen")
        cbar.outline.set_linewidth(1.6)

        # Customize x-axis labels to match quarterly time points
        time_points = heatmap.shape[1]
        quarterly_labels = self.labels.get('Sheet1', [])

        # Ensure x-tick positions match the original number of time points and center them
        ax.set_xticks(np.arange(time_points) + 0.5)  # Center ticks
        ax.set_xticklabels(quarterly_labels, rotation=90,
                           fontsize=20, fontweight='bold', ha='center')

        ax.tick_params(axis='x', which='both', labelsize=20, width=2, length=8)
        ax.tick_params(axis='y', which='both', labelsize=20, width=2, length=8)
        # Set y-axis ticks for companies and center them
        ax.set_yticks(np.arange(len(company_names)) + 0.5)  # Center ticks
        ax.set_yticklabels(company_names, fontsize=20,
                           fontweight='bold', va='center')

        # Customize the plot boundaries
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(2)  # Strong boundary lines

        # Tighten layout to enhance appearance
        plt.tight_layout(rect=[0, 0, 0.9, 1])  # Leave space for the colorbar on the right

        # Save the heatmap
        filename = "heatmap_reconstruction_error_actual.pdf"
        self._save_plot(filename, output_subdir="reconstruction_data")
        plt.close()
        print(f"Reconstruction error heatmap (actual data) saved to '{filename}'")

    @staticmethod
    def gradient_fill(ax, x, y, color, alpha=0.6):

        z = np.linspace(0, 1, len(y))
        cmap = LinearSegmentedColormap.from_list("custom", [color, "white"], N=256)
        for i in range(len(y) - 1):
            ax.fill_between(x[i:i + 2], y[i:i + 2], color=cmap(z[i]), alpha=alpha)

    def plot_cluster_scatter(self):
        """
        Visualize time series clusters in separate scatter plots for each cluster,
        showing only the cluster centers with their gradient fills and members.
        Each company in the cluster will have a distinct color.
        """
        n_clusters = self.cluster_centers.shape[0]  # Number of clusters
        color_palette = sns.color_palette("husl", n_clusters)  # Vibrant color palette for clusters

        for cluster_idx in range(n_clusters):
            # Create a new figure for each cluster
            plt.figure(figsize=(30, 10))

            # Filter the data points that belong to the current cluster
            cluster_data = self.reduced_data[self.cluster_labels == cluster_idx]

            # Ensure there is at least one company in the cluster
            if cluster_data.shape[0] > 0:
                # Assign a unique color for each company in the cluster
                cluster_colors = [
                    '#90EE90', '#00BFFF', '#FF00FF', '#000000', '#FFC0CB',
                    '#FFD700', '#008080', '#FF0000', '#800080', '#40E0D0',
                    '#ADD8E6', '#FFDAB9', '#E6E6FA', '#36454F', '#F5F5DC'
                ]

                # Get the indices of companies in the current cluster
                cluster_company_indices = np.where(self.cluster_labels == cluster_idx)[0]

                # Plot individual company time series in the cluster with different colors
                legend_proxies = []  # To store proxy artists for legend
                legend_labels = []  # To store corresponding labels
                for idx, company_idx in enumerate(cluster_company_indices):
                    # Get the time series for the company at index company_idx
                    ts = self.reduced_data[company_idx]
                    # Get the company name
                    company_name = self.companies[company_idx]
                    # Plot the time series with the unique color for the company
                    plt.plot(ts, color=cluster_colors[idx], alpha=0.7, linewidth=1.5)
                    # plt.plot(ts[:, 0], color=cluster_colors[idx], alpha=0.7, linewidth=1.5)  # Use the first feature
                    # Add the company name to the legend using scatter
                    plt.scatter(np.arange(len(ts)), ts, color=cluster_colors[idx],
                                edgecolor='black', s=150, zorder=5, linewidths=2)
                    # plt.scatter(np.arange(len(ts[:, 0])), ts[:, 0], color=cluster_colors[idx],
                    #             edgecolor='black', s=150, zorder=5, linewidths=2)

                    # Create a proxy artist for the legend
                    proxy = plt.Line2D([0], [0], marker='o', color='w',
                                       markerfacecolor=cluster_colors[idx],
                                       markersize=np.sqrt(400), markeredgecolor='black',
                                       markeredgewidth=2)
                    legend_proxies.append(proxy)
                    legend_labels.append(company_name)

                # Process the cluster center
                cluster_center = self.cluster_centers[cluster_idx]

                if cluster_center.ndim > 1:
                    cluster_center = cluster_center[:, 0]  # we have 2, use the first feature

                time_steps = np.arange(len(cluster_center))
                smooth_time = np.linspace(0, len(cluster_center) - 1, 500)
                smooth_center = make_interp_spline(time_steps, cluster_center, k=3)(smooth_time)

                # Plot the smoothed cluster center line
                plt.plot(smooth_time, smooth_center, color=color_palette[cluster_idx], linewidth=3,
                         label=f"Cluster {cluster_idx + 1} Center")

                self.gradient_fill(plt.gca(), smooth_time, smooth_center,
                                   color=color_palette[cluster_idx])

                # Scatter actual data points for the cluster center
                plt.scatter(time_steps, cluster_center, color=color_palette[cluster_idx],
                            edgecolor='black', s=150, zorder=5, linewidths=2)

                # Add a custom legend with proxy artists for companies
                plt.legend(handles=legend_proxies, labels=legend_labels, loc='upper center',
                           bbox_to_anchor=(0.5, -0.25), fontsize=25, fancybox=True, frameon=True,
                           facecolor='#f0f0f0', framealpha=0.9, edgecolor='gray', ncol=4)

            # Customize plot aesthetics
            plt.title(f"Cluster {cluster_idx + 1} Trends", fontsize=28, fontweight='bold', pad=20)
            plt.xlabel("Time Steps", fontsize=25, fontweight='bold')
            plt.ylabel("Value", fontsize=25, fontweight='bold')

            # Let y-axis adapt dynamically to the data range (include negative values)
            plt.ylim(bottom=None, top=None)
            plt.yticks(fontsize=22)

            # Let x-axis adapt to the data range dynamically
            plt.xlim(left=None)

            # Customize x-axis labels
            horizontal_labels = self.labels['Sheet1']
            plt.xticks(ticks=np.arange(len(horizontal_labels)), labels=horizontal_labels,
                       rotation=45, fontsize=18)

            # Remove gridlines for a cleaner layout
            plt.gca().spines['top'].set_visible(False)
            plt.gca().spines['right'].set_visible(False)
            plt.gca().spines['left'].set_linewidth(1.2)
            plt.gca().spines['bottom'].set_linewidth(1.2)

            plt.gca().set_facecolor('#f9f9f9')  # Light background color

            # Save the plot for the individual cluster
            plot_filename = f"cluster_{cluster_idx + 1}_time_series_with_insurers.png"
            self._save_plot(plot_filename, output_subdir="time_series_clusters")
            print(f"Cluster {cluster_idx + 1} plot saved to {plot_filename}")





