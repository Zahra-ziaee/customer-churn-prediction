from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"
RESULTS_DIR = PROJECT_ROOT / "results"
FIGURES_DIR = RESULTS_DIR / "figures"

CUSTOMER_CHURN_FILE = RAW_DATA_DIR / "telecom_customer_churn.csv"
ZIPCODE_POPULATION_FILE = RAW_DATA_DIR / "telecom_zipcode_population.csv"
DATA_DICTIONARY_FILE = RAW_DATA_DIR / "telecom_data_dictionary.csv"

PROCESSED_FILE = PROCESSED_DATA_DIR / "processed_churn_data.csv"

TARGET_COLUMN = "Customer Status"
TARGET_BINARY_COLUMN = "Churn"

RANDOM_STATE = 42
TEST_SIZE = 0.2

ID_COLUMNS = [
    "Customer ID",
]

LEAKAGE_COLUMNS = [
    "Churn Category",
    "Churn Reason",
]

LOCATION_COLUMNS_TO_DROP = [
    "Latitude",
    "Longitude",
]