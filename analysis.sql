-- Run against a table named predictions loaded from test_predictions.csv.
-- NTILE uses risk ordering; ties may split across deciles.
WITH ranked AS (
 SELECT *, NTILE(10) OVER (ORDER BY predicted_risk DESC) AS risk_decile
 FROM predictions
), deciles AS (
 SELECT risk_decile, COUNT(*) AS customers,
 SUM(actual_churn) AS churners, AVG(predicted_risk) AS predicted_rate,
 AVG(actual_churn) AS observed_rate
 FROM ranked GROUP BY risk_decile
)
SELECT *, SUM(churners) OVER (ORDER BY risk_decile)*1.0 /
 NULLIF(SUM(churners) OVER (),0) AS cumulative_churn_capture
FROM deciles ORDER BY risk_decile;
