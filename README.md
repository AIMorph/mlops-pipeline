# Adult Income MLOps Pipeline

End-to-end MLOps pipeline: Git + DVC, MLflow, pytest, GitHub Actions, Evidently.

## Dataset
Adult Census Income dataset from OpenML (`fetch_openml("adult", version=2)`).
A random sample of 10,000 rows is used. The task is binary classification:
predict whether a person earns more than $50K per year (`income` = 1).
It has 14 features (6 numeric, 8 categorical) and natural missing values.

## Setup
```bash
git clone https://github.com/AIMorph/mlops-pipeline.git
cd mlops-pipeline
python -m venv venv
venv\Scripts\activate        # Windows (Mac/Linux: source venv/bin/activate)
pip install -r requirements.txt
dvc pull                     # or: python -m src.download_data
```

## Run training
```bash
python -m src.train --config configs/config.yaml
```
Training reads all settings from `configs/config.yaml` and exits with code 1
if the metric thresholds are not met.

## Run experiments
```bash
python -m src.train --config configs/config.yaml --override configs/exp2_rf_300.yaml --run-name exp2_rf_300
python -m src.compare_experiments
mlflow ui
```

## Run tests
```bash
pytest tests/ -v
```

## Drift monitoring
```bash
python -m src.monitor_drift
```
Saves an HTML report to `reports/` and exits with code 1 if drift exceeds the
threshold. See `MONITORING.md` for the analysis.

## Structure
- `src/` source code (preprocess, train, evaluate, compare, monitor)
- `configs/` YAML configs
- `tests/` pytest suite
- `.github/workflows/ci.yml` CI/CD pipeline
- `data/raw/adult.csv.dvc` DVC pointer to the dataset

## Preprocessing (no data leakage)
The raw data is split into train and test sets first. All preprocessing lives
inside one scikit-learn `Pipeline` (`ColumnTransformer` with `SimpleImputer` and
`OneHotEncoder(handle_unknown="ignore")`), so medians and categories are learned
from the training rows only and then applied unchanged to the test rows.

## Experiment results
| Run | Model | Key parameters | Accuracy | F1 | ROC AUC |
|---|---|---|---|---|---|
| exp5_gb_200 | gradient_boosting | n_estimators=200, learning_rate=0.05, max_depth=4 | 0.8730 | 0.6962 | 0.9232 |
| exp4_gb_100 | gradient_boosting | n_estimators=100, learning_rate=0.1, max_depth=3 | 0.8685 | 0.6835 | 0.9188 |
| exp2_rf_300 | random_forest | n_estimators=300, max_depth=20 | 0.8645 | 0.6658 | 0.9188 |
| exp1_rf_100 | random_forest | n_estimators=100, max_depth=10 | 0.8610 | 0.6499 | 0.9107 |
| exp3_logreg | logistic_regression | max_iter=1000, C=1.0 | 0.8475 | 0.6373 | 0.9028 |

Best run: **exp5_gb_200** (accuracy 0.8730)
