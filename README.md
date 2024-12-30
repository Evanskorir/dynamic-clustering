# Decoding Financial Health in Kenya’s Medical Insurance Sector: A Data-Driven Cluster Analysis
This repository presents a data-driven analysis of the performance of medical insurance providers in Kenya. By evaluating key financial ratios, including Market Share, Claims Paid Ratio, Claims Incurred Ratio, and Underwriting Profit Ratio, the project aims to assess the financial health and operational efficiency of different insurers.

Using analytical techniques such as Principal Component Analysis (PCA), Autoencoders, and time-series analysis, the project identifies trends and patterns in the data that reflect the insurers' performance over time. These insights may assist policyholders in understanding the relative strengths and weaknesses of various insurance companies, as well as provide valuable information for stakeholders seeking to improve the sector's performance.

Ultimately, this analysis seeks to inform decision-making for both consumers and industry participants, helping to improve transparency and competition in the Kenyan medical insurance market.

## Data Files
```
data
  ├── Medical_quarterly # Quarterly recorded medical insurance data for several medical providers in Kenya
  ├── Medical_yearly    # Yearly recorded medical insurance data for several medical providers in Kenya
  ├── quarterly_labels  # labels for the Medical_quarterly 
  └── yearly_labels     # labels for the Medical_yearly 
```

## Objective
- **Performance Evaluation**
  - **Description**: Analyzing the quarterly performance of medical insurance providers in Kenya using data from the 
  Insurance Regulatory Authority (IRA) website: https://ira.go.ke/. The aim is to assess key financial indicators and 
  trends to understand the strengths and weaknesses of different insurers.
  
## Method
A summary about the steps of the research project:
![Flowchart of Project Methodology](Flowchart.png)


## Folder Structure
```
data                
src                    
 ├── clustering_technique        
 │   └── dynamic_time_warping          
 ├── dimension_reduction       
 │   ├──  autoencoder
 │   └── pca
 ├── evaluation     
 │   ├── elbow_method
 │   │   └── silhouette
 ├── controller  
 ├── dataloader 
 ├── distance_matrix 
 ├── plotter  
 ├── ratios
 ├── ratios_scaler         
 └── reconstruction
main 
README
```

## File Details
#### `src/clustering_technique/`
- **`dynamic_time_warping.py`**: Performs clustering using TimeSeriesKMeans
#### `src/dimension_reduction/`
- **`autoencoder.py`**: Uses autoencoder for dimension reduction.
- **`pca.py`**: Uses PCA for dimension reduction.

#### `src/evaluation/`
- **`elbow_method.py`**: Evaluates the desired number of clusters using elbow method  
- **`silhouette.py`**: Evaluates the desired number of clusters using silhouette score.
#### `src/`
- **`controller.py`**:  Handles data loading and initialization.
- **`dataloader.py`**: Designed to load and preprocess data required for the project.
- **`distance_matrix.py`**: Calculates the distance matrix of the reduced ratio data.
- **`plotter.py`**: Generates a reduced scatter plot to visualize the clusters. It also plots a time series clusters 
and the members. Plots also the distance matrix, the reconstruction errors from the autoencoder approach and 
evaluation method.
- **`ratios.py`**: Uses the loaded data to compute the ratios; Market Share, Claims Paid Ratio,
Claims Incurred Ratio, Underwriting Profit Ratio which are used for the analysis in this project.
- **`ratios_scaler.py`**: Cleans and scales the ratio data using MinMaxScaler() or StandardScale methods.
- **`reconstruction_error.py`**: Calculates the reconstructed data from the autoencoder approach and then 
calculates reconstruction error, then save the results in an Excel file.
- **`main.py`**: Imports and initializes DataLoader to load time series data, then imports Controller, 
initializes it, and run the analysis.

## Implementation

To run the project, follow these steps:
1. Open `main.py` and configure the data type, dimension reduction approach 
and whether to include reinsurers. 
2. Run the code with these steps:
#### Specify the data type for the analysis e.g.
```data_type = "quarterly" ```
#### Load the necessary data for the analysis
```data = DataLoader(data_type=data_type, include_reinsurers=include_reinsurers) ``` 
#### Load the calculated ratio data for the analysis
``` ratios_data = InsuranceRatios(data=data.medical_data, include_reinsurers=include_reinsurers) ```
#### Initialize the Controller with the loaded data
``` analysis_controller = InsuranceAnalysisController(data=ratios_data, data_type=data_type, labels=data.labels, include_reinsurers=include_reinsurers)  ```
#### run the analysis
```analysis_controller.run_analysis(reduction_method=reduction_method) ```

## Output

```
reconstruction
     │    ├── reconstructed_data.csv # Saves the reconstructed data from the autoencoder approach in the Excel file.
     │    └── reconstruction_errors.csv  # Calculates and saves the reconstruction errors from the reconstructed data 
     in the Excel file.
plots
     ├── distance_matrix
     │    ├── ordered_dtw_distance_matrix.png # Visualizes the distance matrix computed using DTW distance.
     │    └── unordered_dtw_distance_matrix.png # Visualizes the ordered distance matrix based on hierarchical
clustering using the complete linkage method.
     ├──  evaluation
     │    ├── elbow_method.png # Visualizes the optimal number of clusters for the study by considering elbow method.
     │    └── silhouette.png  # Visualizes the optimal number of clusters for the study by considering silhouette curve. 
     ├── projection
     │    └── dimension_reduction_reduced_2d_avg_plot.png  # Visualizes the clusters in 2 dimension by 
considering average distance of each insurer.
     ├── time_series_clusters
     │    └── cluster_time_series_with_members.png # Visualizes the time series clusters along with the members.
     ├── time_series_data 
     │    ├── Claims_Incurred_Ratio
     │    │   ├── time_series_average_performing_insurers
     │    │   ├── time_series_best_performing_insurers
     │    │   └── time_series_least_performing_insurers
     │    └── time_series_average_performing_insurers
     │    ├── Claims_Paid_Ratio
     │    │   ├── time_series_average_performing_insurers
     │    │   ├── time_series_best_performing_insurers
     │    │   └── time_series_least_performing_insurers
     │    ├── Market Share
     │    │   ├── time_series_average_performing_insurers
     │    │   ├── time_series_best_performing_insurers
     │    │   └── time_series_least_performing_insurers
     │    ├── Underwriting Profits Ratio 
     │    │   ├── time_series_average_performing_insurers
     │    │   ├── time_series_best_performing_insurers
     │    │   └── time_series_least_performing_insurers

```

## Requirement
This project is developed and tested with Python 3.8 or higher. Install dependencies from `requirements.txt`:
```bash
pip install -r requirements.txt
