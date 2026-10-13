from pathlib import Path

import mlflow
import pandas as pd
import yaml

SKIP_PARAMS = {
    "params.model_type",
    "params.test_size",
    "params.data_version",
    "params.random_state",
}


def key_params(row):
    items = []
    for col in row.index:
        if col.startswith("params.") and col not in SKIP_PARAMS and pd.notna(row[col]):
            items.append(f"{col.replace('params.', '')}={row[col]}")
    return ", ".join(items)


def main():
    with open("configs/config.yaml") as f:
        config = yaml.safe_load(f)

    experiment = mlflow.get_experiment_by_name(config["mlflow"]["experiment_name"])
    runs = mlflow.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=["metrics.accuracy DESC"],
    )

    columns = [
        "run_id",
        "tags.mlflow.runName",
        "params.model_type",
        "metrics.accuracy",
        "metrics.f1",
        "metrics.roc_auc",
    ]
    print(f"Total runs: {len(runs)}")
    print(runs[columns].to_string(index=False))

    best = runs.iloc[0]
    print("\nBEST RUN (by accuracy)")
    print(f"Run name : {best['tags.mlflow.runName']}")
    print(f"Run ID   : {best['run_id']}")
    print(f"Model    : {best['params.model_type']}")
    print(f"Accuracy : {best['metrics.accuracy']:.4f}")

    lines = [
        "| Run | Model | Key parameters | Accuracy | F1 | ROC AUC |",
        "|---|---|---|---|---|---|",
    ]
    for _, row in runs.iterrows():
        lines.append(
            f"| {row['tags.mlflow.runName']} | {row['params.model_type']} "
            f"| {key_params(row)} | {row['metrics.accuracy']:.4f} "
            f"| {row['metrics.f1']:.4f} | {row['metrics.roc_auc']:.4f} |"
        )
    lines.append("")
    lines.append(
        f"Best run: **{best['tags.mlflow.runName']}** "
        f"(accuracy {best['metrics.accuracy']:.4f})"
    )

    out = Path("reports/experiment_summary.md")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\nSummary table saved to {out}")


if __name__ == "__main__":
    main()