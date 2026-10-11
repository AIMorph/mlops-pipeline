import pandas as pd
import pytest
import yaml


@pytest.fixture(scope="module")
def config():
    with open("configs/config.yaml") as f:
        return yaml.safe_load(f)


@pytest.fixture(scope="module")
def df(config):
    return pd.read_csv(config["data"]["path"])


def test_expected_columns_present(df, config):
    d = config["data"]
    expected = set(d["numeric_features"] + d["categorical_features"] + [d["target"]])
    assert expected.issubset(set(df.columns))


def test_target_contains_only_expected_values(df, config):
    assert set(df[config["data"]["target"]].unique()) <= {0, 1}


def test_numeric_features_within_expected_ranges(df):
    assert df["age"].between(17, 90).all()
    assert df["hours_per_week"].between(1, 99).all()
    assert df["education_num"].between(1, 16).all()


def test_dataset_meets_size_requirements(df, config):
    n_features = len(df.columns) - 1
    assert len(df) >= 1000
    assert n_features >= 8


def test_dataset_contains_missing_values(df):
    assert df.isna().sum().sum() > 0