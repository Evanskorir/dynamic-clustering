import streamlit as st
import os
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import scipy.cluster.hierarchy as sch

from dashboard import DashboardController
from src.evaluation.silhouette import ClusteringEvaluation
from src.clustering_technique.dynamic_time_warping import DTWClustering
from src.controller import InsuranceAnalysisController

st.set_page_config(page_title="Insurance Dashboard", layout="wide")

# Custom CSS styling
st.markdown("""
    <style>
    body, .stApp {
        background-color: #e6f2ea;
    }
    .big-metric-box {
        background-color: #c1e1c1;
        padding: 1.5rem;
        border-radius: 12px;
        text-align: center;
        font-size: 1.2rem;
        font-weight: bold;
        color: #1b4332;
        box-shadow: 2px 2px 6px rgba(0,0,0,0.1);
    }
    .main-title {
        font-size: 2rem;
        font-weight: 800;
        color: #1b4332;
    }
    .metric-tile {
        background-color: #ffffff;
        padding: 1rem 1.5rem;
        border-radius: 10px;
        text-align: center;
        font-weight: bold;
        color: #1b4332;
        box-shadow: 2px 2px 8px rgba(0,0,0,0.15);
    }
    .metric-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
        gap: 1rem;
        margin-top: 1rem;
        margin-bottom: 2rem;
    }
    .metric-title {
        font-size: 0.85rem;
        font-weight: 500;
        color: #666;
    }
    .metric-value {
        font-size: 1.4rem;
        font-weight: 800;
        margin-top: 0.25rem;
    }
    </style>
""", unsafe_allow_html=True)

# Sidebar Controls
st.sidebar.title("Insurance Clustering Controls")
reduction_method = st.sidebar.selectbox("Dimensionality Reduction",
                                        ["lstm", "autoencoder", "pca"])
include_reinsurers = st.sidebar.checkbox("Include Reinsurers?", value=False)
n_clusters = st.sidebar.slider("Number of Clusters", 1, 10, 4)
colormap_options = ["jet", "viridis", "plasma", "magma", "cividis", "Reds",
                    "Blues", "coolwarm"]
selected_cmap = st.sidebar.selectbox("Select Heatmap Color Scheme", colormap_options)

# Initialize controller
if "controller" not in st.session_state or \
   st.session_state.get("n_clusters") != n_clusters or \
   st.session_state.get("reduction_method") != reduction_method or \
   st.session_state.get("include_reinsurers") != include_reinsurers or \
   st.sidebar.button("Run Analysis"):

    st.session_state.n_clusters = n_clusters
    st.session_state.reduction_method = reduction_method
    st.session_state.include_reinsurers = include_reinsurers

    st.session_state.controller = DashboardController(
        include_reinsurers=include_reinsurers,
        reduction_method=reduction_method,
        n_clusters=n_clusters
    )

controller = st.session_state.get("controller", None)

if controller:
    # st.title("\U0001F4CA Insurance Dashboard")
    cluster_labels = controller.get_cluster_labels()
    companies = controller.get_companies()
    data_loader = controller.data_loader
    quarterly_data = controller.ratios_data.ratios_data
    quarterly_labels = data_loader.quarterly_labels["Sheet1"]

    variable_names = [
        "Market Share", "Claims Paid Ratio", "Claims Incurred Ratio",
        "Underwriting Profits Ratio", "Expense Ratio", "Combined Ratio",
        "Claims Payout Ratio"
    ]
    variable_time_series = [
        "Gross Premium Income", "Claims Paid",
        "Claims Incurred", "Underwriting Profit"
    ]
    variable_index = {name: i for i, name in enumerate(variable_names)}
    all_companies = list(quarterly_data.keys())

    st.sidebar.markdown("### Data Selection")
    selected_companies = st.sidebar.multiselect("Select Companies",
                                                all_companies, default=all_companies[:5])
    selected_variables = st.sidebar.multiselect("Select Ratios",
                                                variable_names, default=["Market Share"])
    selected_time_series_variables = st.sidebar.multiselect(
        "Select Time Series Variables", variable_time_series,
        default=["Gross Premium Income"])
    selected_range = st.sidebar.slider("Select Quarter Range", 0,
                                       len(quarterly_labels) - 1,
                                       (0, len(quarterly_labels) - 1))
    selected_quarters = quarterly_labels[selected_range[0]:selected_range[1] + 1]

    summary_var = selected_variables[0] if \
        selected_variables else variable_names[0]
    var_idx = variable_index[summary_var]

    top_company = None
    top_value = float('-inf')
    total_value = 0
    valid_company_count = 0

    for company in selected_companies:
        values = quarterly_data[company][
                 selected_range[0]:selected_range[1] + 1, var_idx] \
            if company in quarterly_data else np.array([])
        avg = np.nanmean(values) if len(values) > 0 else 0
        total_value += avg
        valid_company_count += 1
        if avg > top_value:
            top_value = avg
            top_company = company

    avg_value = total_value / valid_company_count if \
        valid_company_count > 0 else 0
    quarter_range_label = f"{quarterly_labels[selected_range[0]]} – " \
                          f"{quarterly_labels[selected_range[1]]}"

    st.markdown(f"""
    <div class="metric-grid">
        <div class="metric-tile" style="background-color: #d4edda;">  <!-- Green -->
            <div class="metric-title">Top Company ({summary_var})</div>
            <div class="metric-value">{top_company}<br>
            <span style='font-size: 0.9rem; color:#444;'>({top_value:,.2f})</span></div>
        </div>
        <div class="metric-tile" style="background-color: #f8d7da;">  <!-- Red -->
            <div class="metric-title">Avg {summary_var} (Selected)</div>
            <div class="metric-value">{avg_value:,.2f}</div>
        </div>
        <div class="metric-tile" style="background-color: #fff3cd;">  <!-- Yellow -->
            <div class="metric-title">Quarter Range</div>
            <div class="metric-value">{quarter_range_label}</div>
        </div>
        <div class="metric-tile" style="background-color: #d1ecf1;">  <!-- Blue -->
            <div class="metric-title">Clusters</div>
            <div class="metric-value">{n_clusters}</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ===============================
    # 📅 Time Series Heatmaps
    # ===============================
    st.markdown("## 📅 Time Series Data")
    plotter = controller.get_plotter()
    quarterly_data = data_loader.quarterly_medical_data
    quarterly_labels = data_loader.quarterly_labels["Sheet1"]

    quarterly_cols = ["Gross Premium Income", "Claims Paid",
                      "Claims Incurred", "Underwriting Profit"]

    for header in selected_time_series_variables:
        fig = plotter.generate_yearly_heatmap_figure(
            yearly_data=quarterly_data,
            quarterly_labels=quarterly_labels,
            header=header.strip(),
            cmap=selected_cmap,
            selected_companies=selected_companies,
            use_quarterly_labels=True,
            column_names=quarterly_cols
        )
        if fig:
            st.markdown(f"#### {header}")
            st.pyplot(fig)
        else:
            st.warning(f"Could not generate heatmap for '{header}'")

    # ===============================
    # 🧱 Distance Matrix Visualizations
    # ===============================
    st.markdown("## 🧱 Distance Matrix Visualizations")
    unordered_dm = controller.get_distance_matrix()
    cont_analysis = controller.analysis.plotter
    ordered_dm, order_idx = cont_analysis._reorder_distance_matrix(unordered_dm)

    col_dm1, col_dm2 = st.columns(2)
    with col_dm1:
        fig, ax = plt.subplots(figsize=(12, 10))
        sns.heatmap(unordered_dm, cmap=selected_cmap, xticklabels=companies,
                    yticklabels=companies, ax=ax, cbar_kws={'label': 'DTW Distance'})
        ax.set_title("Unordered Distance Matrix")
        st.pyplot(fig)

    with col_dm2:
        fig, ax = plt.subplots(figsize=(12, 10))
        ordered_labels = np.array(companies)[order_idx]
        sns.heatmap(ordered_dm, cmap=selected_cmap, xticklabels=ordered_labels,
                    yticklabels=ordered_labels, ax=ax, cbar_kws={'label': 'DTW Distance'})
        ax.set_title("Ordered Distance Matrix")
        st.pyplot(fig)

    # ===============================
    # 📈 Variable Trends
    # ===============================
    st.markdown("## 📈 Variable Trends Over Time")
    col_heatmap, col_ranking = st.columns([5.5, 1.5])

    # ===============================
    # 📊 Evaluation Curves
    # ===============================
    st.markdown("## 📊 Evaluation Curves")
    eval_method = st.selectbox("Select Evaluation Method",
                               ["Elbow", "Silhouette"], index=0)

    if eval_method == "Elbow":
        st.subheader("📉 Elbow Method for Optimal Clusters")
        plotter.plot_elbow(controller.get_reduced_data())
        st.image("./plots/evaluation/elbow_method.png")

    elif eval_method == "Silhouette":
        st.subheader("🧩 Silhouette Scores for Cluster Evaluation")
        plotter.plot_silhouette_curve(controller.get_reduced_data(),
                                      max_clusters=12, metric="dtw")
        st.image("./plots/evaluation/silhouette.png")

    # ===============================
    # 🔀 Cluster Visualizations
    # ===============================
    st.markdown("## 🔀 Cluster Visualizations")
    visualization_mode = st.selectbox("Select Cluster View",
                                      ["Time Series K-Means",
                                       "Hierarchical Dendrogram"])

    if visualization_mode == "Time Series K-Means":
        st.subheader("📈 Cluster Time Series")
        cluster_ids = [f"Cluster {i + 1}" for i in range(len(np.unique(cluster_labels)))]
        selected_cluster = st.selectbox("Select Cluster", cluster_ids)
        cluster_index = int(selected_cluster.split(" ")[-1]) - 1
        reduced_data = controller.get_reduced_data()
        cluster_plot_path = f"./plots/time_series_clusters/" \
                            f"cluster_{cluster_index + 1}_time_series_with_insurers.png"

        plotter.plot_cluster_scatter(approach=reduction_method)
        if os.path.exists(cluster_plot_path):
            st.image(cluster_plot_path, use_container_width=True)
        else:
            st.warning("Cluster time series plot not found.")

    elif visualization_mode == "Hierarchical Dendrogram":
        st.subheader("🌿 Hierarchical Clustering")
        fig, ax = plt.subplots(figsize=(18, 10), dpi=300)
        threshold = InsuranceAnalysisController.get_cluster_threshold_from_linkage(
            controller.analysis.linkage_matrix, n_clusters
        )
        dendro = sch.dendrogram(
            controller.analysis.linkage_matrix, labels=companies,
            color_threshold=threshold, leaf_rotation=90,
            leaf_font_size=14, above_threshold_color='black', ax=ax
        )
        for tick in ax.get_xticklines(): tick.set_visible(False)
        xlbls = ax.get_xmajorticklabels()
        for lbl in xlbls:
            leaf_text = lbl.get_text()
            lbl.set_color(dendro['leaves_color_list'][dendro['ivl'].index(leaf_text)])
            lbl.set_fontweight("medium")

        ax.set_ylabel('Cluster Distance', fontsize=25, fontweight="bold")
        ax.tick_params(axis='x', labelsize=18)
        ax.tick_params(axis='y', labelsize=20)
        ax.spines[['top', 'right', 'bottom']].set_visible(False)
        st.pyplot(fig)
