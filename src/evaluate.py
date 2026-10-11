from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def compute_metrics(model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    return {
        "accuracy": accuracy_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_proba),
    }


def check_thresholds(metrics, thresholds):
    failures = []
    if metrics["accuracy"] < thresholds["min_accuracy"]:
        failures.append(
            f"accuracy {metrics['accuracy']:.4f} < {thresholds['min_accuracy']}"
        )
    if metrics["f1"] < thresholds["min_f1"]:
        failures.append(f"f1 {metrics['f1']:.4f} < {thresholds['min_f1']}")
    return failures