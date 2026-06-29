from pathlib import Path
import requests
import pandas as pd

BASE_DIR = Path(__file__).resolve().parents[1]

PRODUCTION_DATA_PATH = BASE_DIR / "data" / "processed" / "production_normal.csv"

API_URL = "http://127.0.0.1:8000/predict"

df = pd.read_csv(PRODUCTION_DATA_PATH)

# Remove target column because API only needs features
if "y" in df.columns:
    df = df.drop(columns=["y"])

# Send first 30 examples to the API
sample_df = df.head(30)

for idx, row in sample_df.iterrows():
    input_data = row.to_dict()

    response = requests.post(API_URL, json=input_data)

    print(f"Row {idx}: Status {response.status_code}")
    print(response.json())
