import argparse
import sys

import mlflow
import mlflow.sklearn
import pandas as pd
import yaml
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.evaluate import check_thresholds, compute_metrics
from src.preprocess import preprocess


def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)


def get_data_version(dvc_file):
    try:
        with open(dvc_file) as f:
            meta = yaml.safe_load(f)
        return meta["outs"][0]["md5"]
    except FileNotFoundError:
        return "unversioned"


def build_model(model_cfg):
    model_type = model_cfg["type"]
    params = model_cfg.get("params", {})
    if model_type == "random_forest":
        clf = RandomForestClassifier(**params)
    elif model_type == "logistic_regression":
        clf = LogisticRegression(**params)
    elif model_type == "gradient_boosting":
        clf = GradientBoostingClassifier(**params)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    return Pipeline([("scaler", StandardScaler()), ("model", clf)])


def prepare_data(config, sample_size=None):
    d = config["data"]
    df = pd.read_csv(d["path"])
    if sample_size:
        df = df.sample(n=sample_size, random_state=d["random_state"])
    y = df[d["target"]]
    X = preprocess(
        df.drop(columns=[d["target"]]),
        d["numeric_features"],
        d["categorical_features"],
    )
    return train_test_split(
        X, y, test_size=d["test_size"], random_state=d["random_state"], stratify=y
    )


def run_training(config, run_name=None):
    X_train, X_test, y_train, y_test = prepare_data(config)
    model = build_model(config["model"])
    data_version = get_data_version(config["data"]["dvc_file"])

    mlflow.set_experiment(config["mlflow"]["experiment_name"])
    with mlflow.start_run(run_name=run_name):
        mlflow.log_param("model_type", config["model"]["type"])
        mlflow.log_params(config["model"].get("params", {}))
        mlflow.log_param("test_size", config["data"]["test_size"])
        mlflow.log_param("data_version", data_version)
        mlflow.set_tag("data_version", data_version)

        model.fit(X_train, y_train)
        metrics = compute_metrics(model, X_test, y_test)
        mlflow.log_metrics(metrics)
        mlflow.sklearn.log_model(model, "model")
    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--override", default=None)
    parser.add_argument("--run-name", default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    if args.override:
        config.update(load_config(args.override))

    metrics = run_training(config, args.run_name)
    for name, value in metrics.items():
        print(f"{name}: {value:.4f}")

    failures = check_thresholds(metrics, config["thresholds"])
    if failures:
        print("THRESHOLD FAILURES: " + "; ".join(failures))
        sys.exit(1)
    print("All thresholds met.")


if __name__ == "__main__":
    main()