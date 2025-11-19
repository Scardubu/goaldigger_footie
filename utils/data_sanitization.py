import pandas as pd


def sanitize_for_viz(df: pd.DataFrame, expected_columns=None):
    if expected_columns is not None:
        missing = set(expected_columns) - set(df.columns)
        if missing:
            raise ValueError(f"Missing columns for visualization: {missing}")
    # Add additional type checks etc. as needed
    return df.dropna()