import numpy as np
import pytest
import yaml

from src.train import build_model, prepare_data


@pytest.fixture(scope="module")
def trained():
    with open("configs/config.yaml") as f:
        config = yaml.safe_load(f)
    config["model"] = {
        "type": "random_forest",
        "params": {"n_estimators": 50, "max_depth": 10, "random_state": 42},
    }
    d = config["data"]
    X_train, X_test, y_train, y_test = prepare_data(config, sample_size=3000)
    model = build_model(
        config["model"], d["numeric_features"], d["categorical_features"]
    )
    model.fit(X_train, y_train)
    return model, X_test, y_test


def test_predictions_have_correct_type_and_shape(trained):
    model, X_test, _ = trained
    preds = model.predict(X_test)
    assert isinstance(preds, np.ndarray)
    assert preds.shape == (len(X_test),)
    assert set(np.unique(preds)) <= {0, 1}


def test_model_meets_minimum_accuracy(trained):
    model, X_test, y_test = trained
    accuracy = (model.predict(X_test) == y_test).mean()
    assert accuracy >= 0.78