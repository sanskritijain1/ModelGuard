from pathlib import Path
from typing import Dict, Any

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException

from app.database import init_db, log_prediction

BASE_DIR = Path(__file__).resolve().parents[1]

MODEL_PATH = BASE_DIR / "models" / "model.pkl"
TRAIN_DATA_PATH = BASE_DIR / "data" / "processed" / "train.csv"

MODEL_VERSION = "random_forest_v1"

app = FastAPI(
    title="ModelGuard Prediction API",
    description="Prediction API for the Bank Marketing classification model.",
    version="1.0.0",
)

if not MODEL_PATH.exists():
    raise FileNotFoundError(f"Model not found at: {MODEL_PATH}")

if not TRAIN_DATA_PATH.exists():
    raise FileNotFoundError(f"Training data not found at: {TRAIN_DATA_PATH}")

model = joblib.load(MODEL_PATH)

train_df = pd.read_csv(TRAIN_DATA_PATH)
FEATURE_COLUMNS = train_df.drop(columns=["y"]).columns.tolist()

init_db()


@app.get("/")
def home():
    return {
        "message": "ModelGuard API is running",
        "model_version": MODEL_VERSION,
        "features_required": FEATURE_COLUMNS,
    }


@app.post("/predict")
def predict(input_data: Dict[str, Any]):
    try:
        input_df = pd.DataFrame([input_data])

        # Match the exact feature order used during training
        input_df = input_df.reindex(columns=FEATURE_COLUMNS)

        if input_df.isnull().any().any():
            missing_cols = input_df.columns[input_df.isnull().any()].tolist()
            raise HTTPException(
                status_code=400,
                detail=f"Missing values for required columns: {missing_cols}",
            )

        prediction = model.predict(input_df)[0]
        probability_yes = model.predict_proba(input_df)[0][1]

        log_prediction(
            input_data=input_data,
            prediction=prediction,
            probability_yes=probability_yes,
            model_version=MODEL_VERSION,
        )

        return {
            "prediction": int(prediction),
            "prediction_label": "yes" if int(prediction) == 1 else "no",
            "probability_yes": round(float(probability_yes), 4),
            "model_version": MODEL_VERSION,
        }

    except HTTPException:
        raise

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
