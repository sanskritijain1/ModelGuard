from pathlib import Path
import pandas as pd

# Project root = ModelGuard folder
BASE_DIR = Path(__file__).resolve().parents[1]

RAW_DATA_PATH = BASE_DIR / "data" / "raw" / "bank-additional-full.csv"
PROCESSED_DIR = BASE_DIR / "data" / "processed"

PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

if not RAW_DATA_PATH.exists():
    raise FileNotFoundError(f"Dataset not found at: {RAW_DATA_PATH}")

# The Bank Marketing dataset uses ; as separator
df = pd.read_csv(RAW_DATA_PATH, sep=";")

print("Original shape:", df.shape)

# Drop duration because it causes data leakage
if "duration" in df.columns:
    df = df.drop(columns=["duration"])

# Convert target column to 0/1
df["y"] = df["y"].map({"no": 0, "yes": 1})

# Save cleaned full dataset
cleaned_path = PROCESSED_DIR / "bank_cleaned.csv"
df.to_csv(cleaned_path, index=False)

# Time-order split for monitoring simulation
n = len(df)

train_end = int(n * 0.70)
normal_end = int(n * 0.85)

train_df = df.iloc[:train_end]
normal_prod_df = df.iloc[train_end:normal_end]
current_prod_df = df.iloc[normal_end:]

train_df.to_csv(PROCESSED_DIR / "train.csv", index=False)
normal_prod_df.to_csv(PROCESSED_DIR / "production_normal.csv", index=False)
current_prod_df.to_csv(PROCESSED_DIR / "production_current.csv", index=False)

print("Cleaned shape:", df.shape)
print("Train shape:", train_df.shape)
print("Normal production shape:", normal_prod_df.shape)
print("Current production shape:", current_prod_df.shape)

print("\nFiles saved in:", PROCESSED_DIR)
