from pathlib import Path
import json
import pandas as pd

from evidently import Dataset, DataDefinition, Report
from evidently.presets import DataDriftPreset, DataSummaryPreset

BASE_DIR = Path(__file__).resolve().parents[1]

REFERENCE_PATH = BASE_DIR / "data" / "processed" / "train.csv"
CURRENT_PATH = BASE_DIR / "data" / "processed" / "production_current.csv"

REPORTS_DIR = BASE_DIR / "reports"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

HTML_REPORT_PATH = REPORTS_DIR / "drift_report.html"
JSON_REPORT_PATH = REPORTS_DIR / "drift_report.json"
SUMMARY_PATH = REPORTS_DIR / "drift_summary.json"

if not REFERENCE_PATH.exists():
    raise FileNotFoundError(f"Reference data not found: {REFERENCE_PATH}")

if not CURRENT_PATH.exists():
    raise FileNotFoundError(f"Current data not found: {CURRENT_PATH}")

reference_df = pd.read_csv(REFERENCE_PATH)
current_df = pd.read_csv(CURRENT_PATH)

# For drift detection, we monitor input features only.
# The target column y is not available in real production at prediction time.
if "y" in reference_df.columns:
    reference_df = reference_df.drop(columns=["y"])

if "y" in current_df.columns:
    current_df = current_df.drop(columns=["y"])

# Keep the same columns and order
current_df = current_df[reference_df.columns]

numerical_columns = reference_df.select_dtypes(exclude=["object"]).columns.tolist()
categorical_columns = reference_df.select_dtypes(include=["object"]).columns.tolist()

schema = DataDefinition(
    numerical_columns=numerical_columns,
    categorical_columns=categorical_columns,
)

reference_data = Dataset.from_pandas(
    reference_df,
    data_definition=schema,
)

current_data = Dataset.from_pandas(
    current_df,
    data_definition=schema,
)

report = Report(
    [
        DataDriftPreset(),
        DataSummaryPreset(),
    ],
    include_tests=True,
)

my_eval = report.run(
    current_data=current_data,
    reference_data=reference_data,
)

my_eval.save_html(str(HTML_REPORT_PATH))
my_eval.save_json(str(JSON_REPORT_PATH))

# Save a small summary file for Streamlit dashboard
result_dict = my_eval.dict()

summary = {
    "reference_rows": len(reference_df),
    "current_rows": len(current_df),
    "num_features": len(reference_df.columns),
    "numerical_features": numerical_columns,
    "categorical_features": categorical_columns,
    "html_report": str(HTML_REPORT_PATH),
    "json_report": str(JSON_REPORT_PATH),
}

with open(SUMMARY_PATH, "w") as f:
    json.dump(summary, f, indent=4)

print("Drift detection complete.")
print("Reference shape:", reference_df.shape)
print("Current shape:", current_df.shape)
print("Numerical columns:", numerical_columns)
print("Categorical columns:", categorical_columns)
print("Saved HTML report to:", HTML_REPORT_PATH)
print("Saved JSON report to:", JSON_REPORT_PATH)
print("Saved summary to:", SUMMARY_PATH)
