import mlflow
import yaml


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


if __name__ == "__main__":
    main()