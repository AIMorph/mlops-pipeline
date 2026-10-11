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