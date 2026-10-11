# Drift Monitoring Analysis

## Which features showed drift and why?
Drifted features: age, education_num, hours_per_week, workclass (drift share
0.29, above the 0.2 threshold). The "production" data was simulated: age shifted
+12 years, hours_per_week shifted +10, education_num shifted +3, and workclass
was resampled toward "Self-emp-not-inc". This mimics a deployed model receiving
an older, longer-working, more self-employed population than the training data.

## Would this drift likely affect model performance?
Yes. Age, hours worked, and education level are strong predictors of income, so
shifts in their distributions move inputs outside the ranges the model learned.
Predicted probabilities would be less reliable and accuracy would likely drop.

## Recommended action
Investigate first: confirm the shift reflects real changes in incoming data and
not a data pipeline bug. If it is real, retrain on recent data and version the
new dataset with DVC. Keep monitoring after retraining.