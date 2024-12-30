import numpy as np


class InsuranceRatios:
    def __init__(self, data, include_reinsurers=True, claims_threshold=500):
        self.data = data
        self.include_reinsurers = include_reinsurers
        self.claims_threshold = claims_threshold  # claims threshold
        self.ratios_data = {}

        # Threshold limits for various ratios
        self.thresholds = {
            "claims_paid_ratio": claims_threshold,
            "claims_incurred_ratio": claims_threshold,
            "claims_payout_ratio": claims_threshold,
            "underwriting_profit_ratio": 300,
            "combined_ratio": 800,
            "expense_ratio": 800,
        }

        # Separate reinsurers and insurers based on the flag
        self.reinsurers = []
        self.insurers = list(self.data.keys())
        if self.include_reinsurers:
            self.reinsurers = list(self.data.keys())[:5]  # first 5 companies are reinsurers
            self.insurers = list(self.data.keys())[5:]  # Remaining companies are insurers

        # Compute ratios on initialization
        self.compute_ratios()

    def compute_ratios(self):
        np.seterr(divide='ignore', invalid='ignore')

        for company, values in self.data.items():
            gross_premium_income = values[:, 0]
            claims_paid = values[:, 1]
            claims_incurred = values[:, 2]
            underwriting_profits = values[:, 3]
            ratios_per_quarter = []
            for i in range(len(gross_premium_income)):
                if gross_premium_income[i] < 0:
                    print(f"Warning: Negative GPI for {company} at quarter {i}. Setting GPI to 0.")
                    gross_premium_income[i] = 0

                total_gpi_for_quarter = sum([self.data[other_company][i, 0]
                                             for other_company in self.data])

                # Calculate Market Share first
                market_share = (gross_premium_income[
                                    i] / total_gpi_for_quarter) * \
                               100 if total_gpi_for_quarter != 0 else 0

                # Calculate Claims Paid Ratio (without thresholding)
                claims_paid_ratio = self.calculate_ratio(claims_paid[i],
                                                         gross_premium_income[i],
                                                         "claims_paid_ratio")

                # Calculate Claims Incurred Ratio (without thresholding)
                claims_incurred_ratio = self.calculate_ratio(claims_incurred[i],
                                                             gross_premium_income[i],
                                                             "claims_incurred_ratio")

                # Calculate Underwriting Profit Ratio (without thresholding)
                underwriting_profit_ratio = self.calculate_ratio(underwriting_profits[i],
                                                                 gross_premium_income[i],
                                                                 "underwriting_profit_ratio")

                # Calculate Expenses with check to avoid negative or erroneous results
                expenses = gross_premium_income[i] - (claims_incurred[i] + underwriting_profits[i])

                # Check for negative expenses and reset them to 0
                if expenses < 0:
                    print(f"Warning: Negative expenses for {company} at quarter {i}. "
                          f"Setting expenses to 0.")
                    expenses = 0

                # Calculate Expense Ratio (without thresholding)
                expense_ratio = (expenses / gross_premium_income[i]) * \
                                100 if gross_premium_income[i] != 0 else 0

                # Calculate Combined Ratio (without thresholding)
                loss_ratio = claims_incurred_ratio  # Loss ratio is just the claims incurred ratio
                combined_ratio = loss_ratio + expense_ratio

                # Calculate Claims Payout Ratio (without thresholding)
                claims_payout_ratio = self.calculate_ratio(claims_paid[i],
                                                           claims_incurred[i],
                                                           "claims_payout_ratio")

                # Append ratios for this quarter
                ratios_per_quarter.append([
                    market_share,
                    claims_paid_ratio,
                    claims_incurred_ratio,
                    underwriting_profit_ratio,
                    expense_ratio,
                    combined_ratio,
                    claims_payout_ratio,
                ])

            # Store the computed ratios for all quarters for this company
            self.ratios_data[company] = np.array(ratios_per_quarter)

        # After computing all ratios, now apply thresholds
        self.apply_thresholds()

    @staticmethod
    def calculate_ratio(numerator, denominator, ratio_type):
        """
        Helper method to safely calculate a ratio.
        :param numerator: The numerator of the ratio.
        :param denominator: The denominator of the ratio.
        :param ratio_type: The type of the ratio (used for threshold capping).
        :return: Calculated ratio.
        """
        if denominator != 0:
            value = (numerator / denominator) * 100
        else:
            value = 0
        return value

    def apply_thresholds(self):
        """
        Apply the predefined thresholds to all calculated ratios after computation.
        This function ensures the expense and combined ratios are capped correctly.
        """
        for company in self.ratios_data:
            company_ratios = self.ratios_data[company]
            for i in range(len(company_ratios)):
                for j, ratio_value in enumerate(company_ratios[i]):
                    ratio_type = self.get_ratio_type(j)
                    # Skip Market Share from thresholding
                    if ratio_type == "market_share":
                        continue
                    threshold_value = self.thresholds.get(ratio_type, 0)
                    # Apply the threshold
                    if ratio_value > threshold_value:
                        print(f"Threshold applied: {ratio_type} = "
                              f"{ratio_value} capped to {threshold_value}")
                        company_ratios[i, j] = threshold_value

    @staticmethod
    def get_ratio_type(index):
        """
        Maps the index of the ratios to the corresponding ratio name.
        """
        ratio_names = [
            "market_share", "claims_paid_ratio", "claims_incurred_ratio",
            "underwriting_profit_ratio", "expense_ratio",
            "combined_ratio", "claims_payout_ratio"
        ]
        return ratio_names[index]

    def get_results(self):
        """
        Returns the computed ratios for all companies.
        :return: Dictionary containing computed ratios for each company.
        Format: {'Company Name': ndarray with columns [Market Share, Claims Paid Ratio,
        Claims Incurred Ratio, Underwriting Profit Ratio, Expense Ratio,
        Combined Ratio, Claims Payout Ratio]}
        """
        return self.ratios_data
