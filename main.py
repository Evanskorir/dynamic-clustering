from src.controller import InsuranceAnalysisController
from src.dataloader import DataLoader
from src.ratios import InsuranceRatios


def main():
    include_reinsurers = False
    reduction_method = "lstm"
    data = DataLoader(include_reinsurers=include_reinsurers)

    ratios_data = InsuranceRatios(data=data.quarterly_medical_data,
                                  include_reinsurers=include_reinsurers)

    analysis_controller = InsuranceAnalysisController(yearly_data=data.yearly_medical_data,
                                                      data=ratios_data,
                                                      quarterly_labels=data.quarterly_labels,
                                                      yearly_labels=data.yearly_labels,
                                                      include_reinsurers=include_reinsurers)

    analysis_controller.run_analysis(reduction_method=reduction_method)


if __name__ == '__main__':
    main()
