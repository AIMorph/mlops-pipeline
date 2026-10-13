| Run | Model | Key parameters | Accuracy | F1 | ROC AUC |
|---|---|---|---|---|---|
| exp5_gb_200 | gradient_boosting | n_estimators=200, learning_rate=0.05, max_depth=4 | 0.8730 | 0.6962 | 0.9232 |
| exp4_gb_100 | gradient_boosting | n_estimators=100, learning_rate=0.1, max_depth=3 | 0.8685 | 0.6835 | 0.9188 |
| exp2_rf_300 | random_forest | n_estimators=300, max_depth=20 | 0.8645 | 0.6658 | 0.9188 |
| exp1_rf_100 | random_forest | n_estimators=100, max_depth=10 | 0.8610 | 0.6499 | 0.9107 |
| exp3_logreg | logistic_regression | max_iter=1000, C=1.0 | 0.8475 | 0.6373 | 0.9028 |

Best run: **exp5_gb_200** (accuracy 0.8730)
