# Data provenance

IBM Telco Customer Churn sample: https://github.com/IBM/telco-customer-churn-on-icp4d and its data/Telco-Customer-Churn.csv file. The original repository is attributed; raw data are downloaded at runtime, not redistributed. This project code is independently licensed under MIT; third-party source terms remain separate.

## Data dictionary

`customerID`: unique ID, excluded from modeling. `Churn`: binary outcome, Yes/No. `tenure`: months. `MonthlyCharges`, `TotalCharges`: source monetary units; blanks in TotalCharges become missing. `Contract`, `PaymentMethod` and service categories: predictors. `test_predictions.csv` contains only anonymous labels and risk scores. Protected/demographic attributes are excluded from the feature set; exclusion alone does not establish fairness.

## Reproducibility

Source downloads are cached locally and excluded from version control. The pipeline records source content hashes in metrics.json. Reproduce with the pinned requirements and fixed random seeds. Generated summaries may contain transformed aggregates; they do not inherit the project code license when the underlying dataset has separate terms.
