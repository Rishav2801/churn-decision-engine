# Results

## Decision

The validation-selected logistic model reaches test AP 0.625. Contacting the highest-risk 20% captures 46.5% of observed churners at 2.32× prevalence. This supports prioritization research, not a proven retention intervention.

## Economics

Scenario net value = sum(predicted churn risk) × assumed save probability × $100 contribution − contacts × $5. These inputs are hypothetical. Predicting churn does not identify customers whose behavior can be changed. A randomized retention pilot is required.

## Validation

Stratified 60/20/20 split; preprocessing fit only on training; model selected by validation AP; bootstrap intervals conditional on the fixed test predictions. The test set is not used to choose the budget or model.

## Limitations

IBM sample data has no reliable observation/label horizon for prospective deployment. Random splits do not measure future drift. Calibration is assessed, not guaranteed. Permutation importance is associational and unstable among correlated services. Demographics are excluded; this does not establish fairness.
