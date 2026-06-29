from pathlib import Path
import json

import pandas as pd
from scipy.stats import ks_2samp, chi2_contingency

BASE_DIR = Path(__file__).resolve().parents[1]

REFERENCE_PATH = BASE_DIR / "data" / "processed" / "train.csv"
CURRENT_PATH = BASE_DIR / "data" / "processed" / "production_current.csv"

REPORTS_DIR = BASE_DIR / "reports"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_PATH = REPORTS_DIR / "retraining_recommendation.json"

DRIFT_P_VALUE_THRESHOLD = 0.05
DRIFTED_FEATURE_RATIO_THRESHOLD = 0.30


def numerical_drift(reference_series, current_series):
    ref = reference_series.dropna()
    cur = current_series.dropna()

    if len(ref) == 0 or len(cur) == 0:
        return 1.0, False

    _, p_value = ks_2samp(ref, cur)
    drift_detected = p_value < DRIFT_P_VALUE_THRESHOLD

    return float(p_value), drift_detected


def categorical_drift(reference_series, current_series):
    ref_counts = reference_series.value_counts()
    cur_counts = current_series.value_counts()

    all_categories = sorted(set(ref_counts.index).union(set(cur_counts.index)))

    observed = [
        [ref_counts.get(cat, 0) for cat in all_categories],
        [cur_counts.get(cat, 0) for cat in all_categories],
    ]

    try:
        _, p_value, _, _ = chi2_contingency(observed)
        drift_detected = p_value < DRIFT_P_VALUE_THRESHOLD
    except ValueError:
        p_value = 1.0
        drift_detected = False

    return float(p_value), drift_detected


reference_df = pd.read_csv(REFERENCE_PATH)
current_df = pd.read_csv(CURRENT_PATH)

if "y" in reference_df.columns:
    reference_df = reference_df.drop(columns=["y"])

if "y" in current_df.columns:
    current_df = current_df.drop(columns=["y"])

current_df = current_df[reference_df.columns]

feature_results = []

for column in reference_df.columns:
    if reference_df[column].dtype == "object":
        p_value, drift_detected = categorical_drift(
            reference_df[column],
            current_df[column],
        )
        feature_type = "categorical"
    else:
        p_value, drift_detected = numerical_drift(
            reference_df[column],
            current_df[column],
        )
        feature_type = "numerical"

    feature_results.append(
        {
            "feature": column,
            "feature_type": feature_type,
            "p_value": p_value,
            "drift_detected": drift_detected,
        }
    )

total_features = len(feature_results)
drifted_features = sum(item["drift_detected"] for item in feature_results)
drift_ratio = drifted_features / total_features

retraining_recommended = drift_ratio >= DRIFTED_FEATURE_RATIO_THRESHOLD

if retraining_recommended:
    recommendation = (
        "Retraining is recommended because a significant share of input features "
        "show distribution drift compared to the reference training data."
    )
else:
    recommendation = (
        "Retraining is not urgently recommended because the detected drift is below "
        "the chosen monitoring threshold."
    )

summary = {
    "total_features": int(total_features),
    "drifted_features": int(drifted_features),
    "drift_ratio": float(drift_ratio),
    "drift_ratio_threshold": float(DRIFTED_FEATURE_RATIO_THRESHOLD),
    "p_value_threshold": float(DRIFT_P_VALUE_THRESHOLD),
    "retraining_recommended": bool(retraining_recommended),
    "recommendation": recommendation,
    "feature_results": [
        {
            "feature": str(item["feature"]),
            "feature_type": str(item["feature_type"]),
            "p_value": float(item["p_value"]),
            "drift_detected": bool(item["drift_detected"]),
        }
        for item in feature_results
    ],
}

with open(OUTPUT_PATH, "w") as f:
    json.dump(summary, f, indent=4)

print("Retraining recommendation complete.")
print("Total features:", total_features)
print("Drifted features:", drifted_features)
print("Drift ratio:", round(drift_ratio, 3))
print("Retraining recommended:", retraining_recommended)
print("Saved to:", OUTPUT_PATH)
