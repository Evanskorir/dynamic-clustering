import numpy as np

from sklearn.decomposition import PCA


class Pca:
    def __init__(self, scaled_ratios_data):
        # self.scaled_data_list = None
        self.reduced_data = None
        # self.ratios_data = ratios_data
        self.data_scaled = scaled_ratios_data

    def PCA_apply(self, n_components=2):
        """
        Apply PCA to reduce the variables (features) from 4 to a lower dim for each company.
        :param n_components: The number of principal components to retain.
        :return: The reduced data as a 3D numpy array (company's size, time points, reduced features).
        """
        pca = PCA(n_components=n_components, svd_solver='randomized', random_state=50)

        company_results = []

        for company, scaled_data in self.data_scaled.items():
            # Apply PCA to each time point (41 time points with 4 variables)
            pca_result = pca.fit_transform(scaled_data)  # Apply PCA to each company's data (41, 4)
            company_results.append(pca_result)  # Resulting shape for each company: (41, reduced dim)

        # Convert the list of arrays into a 3D numpy array: (company's size, 41, reduced dim)
        self.reduced_data = np.array(company_results)
        return self.reduced_data
