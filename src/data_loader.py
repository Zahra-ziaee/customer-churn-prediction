from pathlib import Path
from typing import Tuple

import pandas as pd

from src.config import (
    CUSTOMER_CHURN_FILE,
    ZIPCODE_POPULATION_FILE,
    DATA_DICTIONARY_FILE,
)


def load_csv_file(file_path: Path) -> pd.DataFrame:
    """
    Load CSV file with encoding fallback.
    Some Maven Analytics files may not load correctly with the default UTF-8 encoding.
    """
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    encodings_to_try = ["utf-8", "utf-8-sig", "latin-1", "cp1252"]

    last_error = None

    for encoding in encodings_to_try:
        try:
            return pd.read_csv(file_path, encoding=encoding)
        except UnicodeDecodeError as error:
            last_error = error

    raise UnicodeDecodeError(
        last_error.encoding,
        last_error.object,
        last_error.start,
        last_error.end,
        f"Could not read {file_path} with tried encodings: {encodings_to_try}",
    )


def load_raw_data() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load raw Maven Analytics Telecom Customer Churn dataset files.
    """
    churn_df = load_csv_file(CUSTOMER_CHURN_FILE)
    zipcode_df = load_csv_file(ZIPCODE_POPULATION_FILE)
    dictionary_df = load_csv_file(DATA_DICTIONARY_FILE)

    return churn_df, zipcode_df, dictionary_df


def print_dataset_summary(df: pd.DataFrame) -> None:
    print("=" * 60)
    print("Dataset Summary")
    print("=" * 60)

    print(f"Rows: {df.shape[0]}")
    print(f"Columns: {df.shape[1]}")

    print("\nColumns:")
    for column in df.columns:
        print(f"- {column}")

    print("\nMissing values:")
    print(df.isna().sum())

    print("\nFirst rows:")
    print(df.head())