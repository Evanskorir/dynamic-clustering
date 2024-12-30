from src.controller import InsuranceAnalysisController
from src.dataloader import DataLoader
from src.ratios import InsuranceRatios


def main():

    data_type = "quarterly"
    include_reinsurers = False
    reduction_method = "autoencoder"
    data = DataLoader(data_type=data_type, include_reinsurers=include_reinsurers)

    ratios_data = InsuranceRatios(data=data.medical_data,
                                  include_reinsurers=include_reinsurers)

    analysis_controller = InsuranceAnalysisController(data=ratios_data,
                                                      data_type=data_type,
                                                      labels=data.labels,
                                                      include_reinsurers=include_reinsurers)
    analysis_controller.run_analysis(reduction_method=reduction_method)


if __name__ == '__main__':
    main()
