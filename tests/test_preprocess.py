import numpy as np
import pandas as pd
import pytest

from src.preprocess import encode_categoricals, fill_missing_values

NUM = ["age"]
CAT = ["workclass"]


@pytest.fixture
def sample_df():
    return pd.DataFrame(
        {
            "age": [25, np.nan, 40, 35],
            "workclass": ["Private", None, "State-gov", "Private"],
            "income": [0, 1, 0, 1],
        }
    )


def test_fill_numeric_missing_uses_median(sample_df):
    out = fill_missing_values(sample_df, NUM, CAT)
    assert out["age"].isna().sum() == 0
    assert out.loc[1, "age"] == 35


def test_fill_categorical_missing_uses_unknown(sample_df):
    out = fill_missing_values(sample_df, NUM, CAT)
    assert out.loc[1, "workclass"] == "Unknown"


def test_encode_creates_dummy_columns(sample_df):
    filled = fill_missing_values(sample_df, NUM, CAT)
    out = encode_categoricals(filled, CAT)
    assert "workclass" not in out.columns
    assert "workclass_Private" in out.columns


def test_encode_leaves_no_text_columns(sample_df):
    filled = fill_missing_values(sample_df, NUM, CAT)
    out = encode_categoricals(filled, CAT)
    assert not any(out[c].dtype == object for c in out.columns)


def test_fill_does_not_modify_original(sample_df):
    original = sample_df.copy()
    fill_missing_values(sample_df, NUM, CAT)
    pd.testing.assert_frame_equal(sample_df, original)


def test_encode_does_not_modify_original(sample_df):
    filled = fill_missing_values(sample_df, NUM, CAT)
    original = filled.copy()
    encode_categoricals(filled, CAT)
    pd.testing.assert_frame_equal(filled, original)


def test_non_dataframe_raises_type_error():
    with pytest.raises(TypeError):
        fill_missing_values([1, 2, 3], NUM, CAT)


def test_missing_column_raises_value_error(sample_df):
    with pytest.raises(ValueError):
        fill_missing_values(sample_df, ["not_a_column"], CAT)


def test_empty_dataframe_raises_value_error():
    with pytest.raises(ValueError):
        fill_missing_values(pd.DataFrame(), NUM, CAT)