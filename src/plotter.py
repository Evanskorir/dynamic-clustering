import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import os
import scipy.cluster.hierarchy as sch
import seaborn as sns
from matplotlib.ticker import ScalarFormatter
from scipy.cluster.hierarchy import fcluster
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
from src.evaluation.silhouette import ClusteringEvaluation
from src.evaluation.elbow_method import ElbowMethod
from scipy.interpolate import make_interp_spline

matplotlib.use('agg')


class Plotter:
    def __init__(self, cluster_centers, time_series_data, data_scaled,
                 cluster_labels, companies, dtw_clustering, labels,
                 include_reinsurers, reduced_data=None, reconstructed_data=None,
                 yearly_cols=None):

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
        self.yearly_cols = yearly_cols

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

    def plot_hierarchical_dendrogram(self, linkage_matrix, threshold=None):
        fig, ax = plt.subplots(figsize=(18, 10), dpi=300)
        cluster_colors = ['#1b9e77', '#d95f02', '#7570b3', '#e7298a',
                          '#66a61e', '#e6ab02', '#a6761d', '#666666',
                          '#1f78b4', '#b2df8a']

        sch.set_link_color_palette(cluster_colors)

        # Generate dendrogram and get leaf label colors
        dendrogram = sch.dendrogram(
            linkage_matrix,
            color_threshold=threshold,
            leaf_rotation=90,
            leaf_font_size=14,
            show_leaf_counts=False,
            labels=self.companies,
            above_threshold_color='black',
            ax=ax
        )

        for tick in ax.get_xticklines():
            tick.set_visible(False)

        # Apply matching colors to leaf labels
        xlbls = ax.get_xmajorticklabels()
        for lbl in xlbls:
            leaf_text = lbl.get_text()
            lbl.set_color(dendrogram['leaves_color_list'][
                              dendrogram['ivl'].index(leaf_text)])
            lbl.set_fontweight("medium")

        ax.tick_params(axis='x', labelsize=18)
        ax.tick_params(axis='y', labelsize=18)
        ax.set_ylabel('Cluster Distance', fontsize=25, fontweight="bold",
                      color="black", labelpad=15)
        ax.tick_params(axis='y', labelsize=20, width=3.5, length=8, colors="black")

        # Clean look: no grid, no threshold shading
        ax.grid(False)
        ax.spines['left'].set_color("black")
        ax.spines['left'].set_linewidth(2)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)

        plt.tight_layout()
        self._save_plot("hierarchical_dendrogram_clean.pdf",
                        output_subdir="hierarchical")

    def plot_time_series_heatmap(self, time_series):
        """
        Plots a heatmap of the reduced time series data for each company.
        """

        if time_series is None:
            raise ValueError("Reduced time series data is not available.")

        # Ensure the input is a NumPy array for consistency
        heatmap_array = np.array(time_series)

        # Remove singleton dimensions (e.g., from shape (28, 41, 1) to (28, 41))
        if heatmap_array.ndim == 3:
            heatmap_array = np.squeeze(heatmap_array)  # Remove the third dimension if it's of size 1

        # Calculate the aspect ratio to maintain a rectangular shape
        aspect_ratio = heatmap_array.shape[1] / heatmap_array.shape[0]
        fig, ax = plt.subplots(figsize=(30, 14))

        # Plot the heatmap using 'viridis' colormap for smooth gradients
        cax = sns.heatmap(
            heatmap_array,
            cmap="jet",
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
        cbar.ax.set_ylabel("Reduced data", fontsize=20, fontweight='bold',
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

    def plot_yearly_time_series_heatmaps(self, yearly_data, yearly_labels, headers, cmap="jet"):
        """
        Plots heatmaps for each selected yearly variable (header), showing company trends over years.
        """
        max_years = 11  # Standardized number of years
        x_labels = [str(int(year)) for year in yearly_labels['Sheet1']]

        for header in headers:
            print(f"Plotting heatmap for: {header}")

            if header not in self.labels['Sheet1']:
                print(f"Warning: '{header}' not found in column labels. Skipping.")
                continue

            header_idx = self.labels['Sheet1'].index(header)
            company_labels, header_data = [], []

            for company in self.companies:
                if company in yearly_data:
                    data = yearly_data[company][:, header_idx]
                    company_labels.append(company)
                else:
                    data = np.zeros(max_years, dtype=float)

                # Ensure fixed length for all rows
                if data.shape[0] < max_years:
                    data = np.pad(data, (0, max_years - data.shape[0]), mode='constant')
                else:
                    data = data[:max_years]

                header_data.append(data)

            header_data = np.array(header_data, dtype=np.float64)

            fig, ax = plt.subplots(figsize=(30, 14))
            cax = sns.heatmap(
                header_data,
                cmap=cmap,
                xticklabels=x_labels,
                yticklabels=company_labels,
                cbar=False,
                linewidths=0,
                ax=ax,
                linecolor='none'
            )

            # Add custom colorbar
            cbar = fig.colorbar(
                cax.collections[0], ax=ax, orientation='vertical',
                fraction=0.03, pad=0.04
            )
            cbar.ax.set_ylabel(
                header.upper(), fontsize=22, fontweight='bold',
                color="darkgreen", labelpad=25
            )
            cbar.ax.tick_params(labelsize=20, colors="darkgreen")
            cbar.outline.set_linewidth(2)

            # Styling
            ax.set_xlabel("")
            ax.set_ylabel("")
            ax.tick_params(axis='x', labelsize=18, width=2, length=8)
            ax.tick_params(axis='y', labelsize=18, width=2, length=8)
            for spine in ax.spines.values():
                spine.set_visible(True)
                spine.set_linewidth(2)
                spine.set_color("black")

            plt.title(f"{header} Over Time", fontsize=24, fontweight='bold',
                      color="black", pad=30)
            plt.tight_layout(rect=[0.15, 0, 0.9, 1])

            filename = f"heatmap_{header.replace(' ', '_').lower()}.png"
            self._save_plot(filename, output_subdir="yearly_time_series_heatmaps")
            plt.close()
            print(f"Heatmap for '{header}' saved to '{filename}'")

    def plot_variable_split2(self, variable_index, variable_name):
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
        heatmap_array = np.array(heatmap_data)

        # Adjust aspect ratio for a more rectangular plot
        aspect_ratio = len(heatmap_array[0]) / len(companies)  # Based on data dimensions
        fig, ax = plt.subplots(figsize=(30, 14))  # Explicitly set a rectangular size

        # Plot the heatmap without a default colorbar
        cax = sns.heatmap(
            heatmap_array,
            cmap="jet",
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
        filename = f"heatmap_{variable_name.replace(' ', '_').lower()}.png"
        self._save_plot(filename, output_subdir=f"heatmaps/{variable_name}")
        plt.close()
        print(f"Heatmap for '{variable_name}' saved to '{filename}'")

    def plot_variable_split(self, variable_index, variable_name):
        """
        Plots a heatmap for the variable across all companies.
        Prints time series data for the specified variable (e.g., Underwriting Profits Ratio).
        """

        # Prepare the data in dictionary format for all variables
        variable_data = {
            company: values[:, variable_index].astype(float)
            for company, values in self.time_series_data.items()
            if values.ndim == 2 and values.shape[1] > variable_index
        }
        for company, series in variable_data.items():
            print(f"  {company}: {np.round(series, 3)}")

        # Determine companies to include
        if self.include_reinsurers:
            companies = list(variable_data.keys())[5:]  # Exclude the first 5 companies
        else:
            companies = list(variable_data.keys())  # Include all companies

        # Gather data into a 2D array
        heatmap_data = [variable_data[company] for company in companies]
        heatmap_array = np.array(heatmap_data)

        # Plotting logic (unchanged from before)
        fig, ax = plt.subplots(figsize=(30, 14))
        cax = sns.heatmap(
            heatmap_array,
            cmap="jet",
            xticklabels=False,
            yticklabels=False,
            cbar=False,
            linewidths=0,
            ax=ax,
            linecolor='none',
        )

        cbar = fig.colorbar(
            cax.collections[0],
            ax=ax,
            orientation='vertical',
            fraction=0.03,
            pad=0.04,
            format='%d%%'
        )
        cbar.ax.set_ylabel(variable_name, fontsize=20, fontweight='bold',
                           color="darkgreen", labelpad=20)
        cbar.ax.tick_params(labelsize=20, colors="darkgreen")
        cbar.outline.set_linewidth(1.5)

        num_time_points = heatmap_array.shape[1]
        quarterly_labels = self.labels.get('Sheet1', [])
        ax.set_xticks(np.arange(num_time_points) + 0.5)
        ax.set_xticklabels(quarterly_labels, rotation=90,
                           fontsize=18, fontweight='bold', ha='center')

        ax.set_yticks(np.arange(len(companies)) + 0.5)
        ax.set_yticklabels(companies, fontsize=18, fontweight='bold', va='center')

        ax.tick_params(axis='x', which='both', labelsize=18, width=2, length=8)
        ax.tick_params(axis='y', which='both', labelsize=18, width=2, length=8)

        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(2)

        plt.tight_layout(rect=[0, 0, 0.9, 1])

        filename = f"heatmap_{variable_name.replace(' ', '_').lower()}.png"
        self._save_plot(filename, output_subdir=f"heatmaps/{variable_name}")
        plt.close()

    def generate_yearly_heatmap_figure(self, yearly_data, quarterly_labels, header,
                                       cmap="jet", selected_companies=None,
                                       column_names=None, use_quarterly_labels=True):
        """
        Generate a heatmap using either quarterly or yearly labels.
        Supports dynamic switching to use high-resolution quarterly labels.
        """

        # === Normalize and lookup variable index ===
        header_clean = " ".join(header.strip().lower().split())

        # Use passed column_names or fallback to self.yearly_cols
        columns = column_names if column_names else self.yearly_cols
        header_lookup = {col.strip().lower(): i for i, col in enumerate(columns)}

        if header_clean not in header_lookup:
            print(f"[ERROR] Header '{header}' not found in yearly columns.")
            return None

        variable_index = header_lookup[header_clean]

        # === Select Labels ===
        raw_labels = quarterly_labels if use_quarterly_labels else self.yearly_labels
        x_labels = list(raw_labels)
        num_periods = len(x_labels)

        if selected_companies is None:
            selected_companies = self.companies

        heatmap_data = []
        final_company_labels = []

        for company in selected_companies:
            values = yearly_data.get(company)

            if values is None or values.ndim != 2 or values.shape[1] <= variable_index:
                print(f"[WARN] Skipping {company}: Invalid shape or missing.")
                continue

            cleaned_rows = []
            for row in values:
                try:
                    float(row[variable_index])
                    cleaned_rows.append(row)
                except (ValueError, TypeError):
                    continue

            if not cleaned_rows:
                print(f"[SKIP] {company}: No valid rows after filtering.")
                continue

            try:
                cleaned_array = np.array(cleaned_rows, dtype=np.float64)
                series = cleaned_array[:, variable_index]

                if len(series) < num_periods:
                    series = np.pad(series, (0, num_periods - len(series)), constant_values=np.nan)
                elif len(series) > num_periods:
                    series = series[:num_periods]

                print(f"  [DEBUG] {company}: len={len(series)} | {np.round(series, 2)}")

                heatmap_data.append(series)
                final_company_labels.append(company)

            except Exception as e:
                print(f"[ERROR] Failed for {company}: {e}")

        if not heatmap_data:
            print("[ERROR] No valid company data to plot.")
            return None

        heatmap_array = np.vstack(heatmap_data)

        fig_height = max(5, len(final_company_labels) * 0.5)
        fig, ax = plt.subplots(figsize=(30, fig_height))

        cax = sns.heatmap(
            heatmap_array,
            cmap=cmap,
            xticklabels=x_labels,
            yticklabels=final_company_labels,
            cbar=False,
            linewidths=0,
            ax=ax,
            linecolor='none'
        )

        cbar = fig.colorbar(
            cax.collections[0], ax=ax, orientation='vertical',
            fraction=0.03, pad=0.04
        )
        cbar.ax.set_ylabel(header.upper(), fontsize=15, color="black", labelpad=25)
        cbar.ax.tick_params(labelsize=18, colors="black")
        cbar.outline.set_linewidth(2)

        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.tick_params(axis='x', labelsize=14, width=2, length=8, rotation=45)
        ax.tick_params(axis='y', labelsize=14, width=2, length=8)

        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(2)
            spine.set_color("black")

        plt.title(f"{header} Over Time", fontsize=24, color="black", pad=30)
        plt.tight_layout(rect=[0.15, 0, 0.9, 1])

        return fig

    def plot_yearly_time_series_heatmaps2(self, yearly_data, yearly_labels,
                                          headers, cmap="jet"):
        """
        Plots heatmaps for each header, visualizing company data over time.
        """
        max_years = 11  # Standardized number of years
        x_labels = [str(int(year)) for year in yearly_labels['Sheet1']]

        for idx, header in enumerate(headers):
            print(f"Plotting heatmap for: {header}")

            # Extract data for the header
            header_data = []
            company_labels = []
            for company in self.companies:
                if company in yearly_data:
                    data = yearly_data[company][:, idx]
                    company_labels.append(company)
                else:
                    data = np.zeros(max_years, dtype=int)

                if data.shape[0] < max_years:
                    padding = np.zeros((max_years - data.shape[0],), dtype=int)
                    data = np.concatenate([data, padding])
                elif data.shape[0] > max_years:
                    data = data[:max_years]

                header_data.append(data)

            header_data = np.array(header_data, dtype=np.float64)

            # Create an annotation array
            annot_data = np.array([[
                "{:.0f}".format(val) if val != 0 else "0" for val in row] for
                row in header_data
            ])

            vmin, vmax = np.min(header_data), np.max(header_data)
            if vmin == vmax:
                vmin, vmax = 0, vmax + max(1, 0.1 * vmax)

            # Plot the heatmap
            # viridis, cividis, magma, plasma, coolwarm, RdBu, PuOr
            fig, ax = plt.subplots(figsize=(30, 14))
            cax = sns.heatmap(
                header_data, cmap=cmap, xticklabels=x_labels,
                yticklabels=company_labels,
                cbar=False, linewidths=0, ax=ax, linecolor='none'
            )

            # Add a custom colorbar
            cbar = fig.colorbar(
                cax.collections[0], ax=ax, orientation='vertical', fraction=0.03, pad=0.04
            )
            cbar.ax.set_ylabel(
                header.upper(), fontsize=22, fontweight='bold',
                color="darkgreen", labelpad=25
            )
            cbar.ax.tick_params(labelsize=20, colors="darkgreen")
            cbar.outline.set_linewidth(2)

            # Remove x-axis and y-axis labels for a professional look
            ax.set_xlabel("")
            ax.set_ylabel("")

            # Customize tick appearance
            ax.tick_params(axis='x', which='both', labelsize=18, width=2, length=8)
            ax.tick_params(axis='y', which='both', labelsize=18, width=2, length=8)

            # Enhance plot boundaries
            for spine in ax.spines.values():
                spine.set_visible(True)
                spine.set_linewidth(2)
                spine.set_color("black")

            plt.title(f"{header} Over Time", fontsize=24, fontweight='bold',
                      color="black", pad=30)
            plt.tight_layout(rect=[0.15, 0, 0.9, 1])  # Adjusted left padding to give space for labels

            filename = f"heatmap_{header.replace(' ', '_').lower()}.png"
            self._save_plot(filename, output_subdir="yearly_time_series_heatmaps")
            plt.close()
            print(f" Heatmap for '{header}' saved to '{filename}'")

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
        # colors = ["darkgreen", "purple", 'red', 'cyan', 'darkgray', 'indigo']
        n_clusters = len(np.unique(self.cluster_labels))
        colors = plt.cm.tab10(np.linspace(0, 1, n_clusters))
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

    def plot_cluster_scatter(self, approach):
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
                # cluster_colors = [
                #     '#90EE90', '#00BFFF', '#FF00FF', '#000000', '#FFC0CB',
                #     '#FFD700', '#008080', '#FF0000', '#800080', '#40E0D0',
                #     '#ADD8E6', '#FFDAB9', '#E6E6FA', '#36454F', '#F5F5DC'
                # ]
                num_companies = len(self.companies)
                cluster_colors = sns.color_palette("husl", num_companies)

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
                    if approach == "lstm":
                        plt.scatter(np.arange(len(ts)), ts, color=cluster_colors[idx],
                                    edgecolor='black', s=150, zorder=5, linewidths=2)
                    else:
                        plt.scatter(np.arange(len(ts[:, 0])), ts[:, 0], color=cluster_colors[idx],
                                    edgecolor='black', s=150, zorder=5, linewidths=2)

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





