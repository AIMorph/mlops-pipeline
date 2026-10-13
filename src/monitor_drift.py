import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from evidently.metric_preset import DataDriftPreset
from evidently.report import Report
from sklearn.model_selection import train_test_split

from src.preprocess import validate_input


def simulate_production(df, seed=42):
    rng = np.random.default_rng(seed)
    prod = df.copy()
    prod["age"] = prod["age"] + 12
    prod["hours_per_week"] = prod["hours_per_week"] + 10
    prod["education_num"] = (prod["education_num"] + 3).clip(upper=16)
    prod["workclass"] = rng.choice(
        ["Self-emp-not-inc", "Private", "Unknown"],
        size=len(prod),
        p=[0.6, 0.3, 0.1],
    )
    return prod


def fill_missing_from_reference(reference, current, numeric_cols, categorical_cols):
    ref = reference.copy()
    cur = current.copy()
    for col in numeric_cols:
        median = ref[col].median()
        ref[col] = ref[col].fillna(median)
        cur[col] = cur[col].fillna(median)
    for col in categorical_cols:
        ref[col] = ref[col].fillna("Unknown")
        cur[col] = cur[col].fillna("Unknown")
    return ref, cur


def main():
    with open("configs/config.yaml") as f:
        config = yaml.safe_load(f)

    d = config["data"]
    threshold = config["monitoring"]["drift_threshold"]
    report_path = Path(config["monitoring"]["report_path"])
    features = d["numeric_features"] + d["categorical_features"]

    df = pd.read_csv(d["path"])
    validate_input(df, features)

    reference, holdout = train_test_split(
        df[features], test_size=d["test_size"], random_state=d["random_state"]
    )
    current = simulate_production(holdout)
    reference, current = fill_missing_from_reference(
        reference, current, d["numeric_features"], d["categorical_features"]
    )

    report = Report(metrics=[DataDriftPreset()])
    report.run(reference_data=reference, current_data=current)

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report.save_html(str(report_path))

    result = report.as_dict()
    table = next(
        m["result"] for m in result["metrics"] if m["metric"] == "DataDriftTable"
    )
    drift_by_col = table["drift_by_columns"]
    drifted = [c for c, v in drift_by_col.items() if v["drift_detected"]]
    share = len(drifted) / len(drift_by_col)

    print("DRIFT SUMMARY")
    for col, v in drift_by_col.items():
        status = "DRIFT" if v["drift_detected"] else "ok"
        print(f"  {col:<18} {status:<6} score={v['drift_score']:.4f}")
    print(f"Drifted features: {drifted}")
    print(f"Drift share: {share:.2f} (threshold {threshold})")
    print(f"HTML report saved to {report_path}")

    if share > threshold:
        print("DRIFT THRESHOLD EXCEEDED")
        sys.exit(1)
    print("Drift within threshold.")


if __name__ == "__main__":
    main()