import numpy as np
import pandas as pd
import pytest

from src.preprocess import build_preprocessor, validate_input

NUM = ["age"]
CAT = ["workclass"]


@pytest.fixture
def sample_df():
    return pd.DataFrame(
        {
            "age": [25, np.nan, 40, 35],
            "workclass": ["Private", np.nan, "State-gov", "Private"],
            "income": [0, 1, 0, 1],
        }
    )


def test_numeric_missing_filled_with_median(sample_df):
    out = build_preprocessor(NUM, CAT).fit_transform(sample_df)
    assert out["age"].isna().sum() == 0
    assert out.loc[1, "age"] == 35


def test_categorical_missing_becomes_unknown_column(sample_df):
    out = build_preprocessor(NUM, CAT).fit_transform(sample_df)
    assert "workclass_Unknown" in out.columns
    assert out.loc[1, "workclass_Unknown"] == 1


def test_encoding_creates_dummy_columns(sample_df):
    out = build_preprocessor(NUM, CAT).fit_transform(sample_df)
    assert "workclass" not in out.columns
    assert "workclass_Private" in out.columns


def test_output_has_no_text_columns(sample_df):
    out = build_preprocessor(NUM, CAT).fit_transform(sample_df)
    assert all(pd.api.types.is_numeric_dtype(out[c]) for c in out.columns)


def test_preprocessing_does_not_modify_original(sample_df):
    original = sample_df.copy()
    build_preprocessor(NUM, CAT).fit_transform(sample_df)
    pd.testing.assert_frame_equal(sample_df, original)


def test_imputation_uses_training_statistics_only():
    train = pd.DataFrame(
        {"age": [10, 20, 30, np.nan], "workclass": ["a", "b", "a", "b"]}
    )
    test = pd.DataFrame({"age": [np.nan, 1000, 2000], "workclass": ["a", "b", "a"]})
    preprocessor = build_preprocessor(NUM, CAT).fit(train)
    out = preprocessor.transform(test)
    # Training median is 20. The test-set median (1500) must NOT be used.
    assert out.loc[0, "age"] == 20


def test_unseen_category_is_ignored_at_transform():
    train = pd.DataFrame({"age": [10, 20, 30], "workclass": ["a", "b", "a"]})
    test = pd.DataFrame({"age": [15], "workclass": ["never_seen"]})
    preprocessor = build_preprocessor(NUM, CAT).fit(train)
    out = preprocessor.transform(test)
    assert out.shape[1] == preprocessor.transform(train).shape[1]
    assert out.filter(like="workclass_").sum(axis=1).iloc[0] == 0


def test_non_dataframe_raises_type_error():
    with pytest.raises(TypeError):
        validate_input([1, 2, 3], NUM)


def test_missing_column_raises_value_error(sample_df):
    with pytest.raises(ValueError):
        validate_input(sample_df, ["not_a_column"])


def test_empty_dataframe_raises_value_error():
    with pytest.raises(ValueError):
        validate_input(pd.DataFrame(), NUM)