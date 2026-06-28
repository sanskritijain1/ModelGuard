from pathlib import Path
import json
import joblib
import pandas as pd
import mlflow
import mlflow.sklearn

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)

BASE_DIR = Path(__file__).resolve().parents[1]

DATA_PATH = BASE_DIR / "data" / "processed" / "train.csv"
MODELS_DIR = BASE_DIR / "models"
REPORTS_DIR = BASE_DIR / "reports"

MODELS_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

if not DATA_PATH.exists():
    raise FileNotFoundError(f"Training file not found: {DATA_PATH}")

df = pd.read_csv(DATA_PATH)

X = df.drop(columns=["y"])
y = df["y"]

categorical_features = X.select_dtypes(include=["object"]).columns.tolist()
numeric_features = X.select_dtypes(exclude=["object"]).columns.tolist()

print("Categorical features:", categorical_features)
print("Numeric features:", numeric_features)

X_train, X_val, y_train, y_val = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y,
)

numeric_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ]
)

categorical_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore")),
    ]
)

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features),
    ]
)

models = {
    "logistic_regression": LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        random_state=42,
    ),
    "random_forest": RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    ),
}

mlflow.set_experiment("ModelGuard Bank Marketing")

results = []
best_model = None
best_model_name = None
best_auc = -1

for model_name, model in models.items():
    print(f"\nTraining: {model_name}")

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model),
        ]
    )

    with mlflow.start_run(run_name=model_name):
        pipeline.fit(X_train, y_train)

        y_pred = pipeline.predict(X_val)
        y_prob = pipeline.predict_proba(X_val)[:, 1]

        metrics = {
            "accuracy": accuracy_score(y_val, y_pred),
            "precision": precision_score(y_val, y_pred, zero_division=0),
            "recall": recall_score(y_val, y_pred, zero_division=0),
            "f1": f1_score(y_val, y_pred, zero_division=0),
            "roc_auc": roc_auc_score(y_val, y_prob),
        }

        print(metrics)

        mlflow.log_param("model_name", model_name)
        mlflow.log_param("target", "y")
        mlflow.log_param("dropped_column", "duration")
        mlflow.log_metrics(metrics)

        mlflow.sklearn.log_model(pipeline, "model")

        results.append(
            {
                "model_name": model_name,
                **metrics,
            }
        )

        if metrics["roc_auc"] > best_auc:
            best_auc = metrics["roc_auc"]
            best_model = pipeline
            best_model_name = model_name

results_df = pd.DataFrame(results)
results_df.to_csv(REPORTS_DIR / "model_results.csv", index=False)

joblib.dump(best_model, MODELS_DIR / "model.pkl")

with open(REPORTS_DIR / "best_model.json", "w") as f:
    json.dump(
        {
            "best_model": best_model_name,
            "best_roc_auc": best_auc,
        },
        f,
        indent=4,
    )

print("\nTraining complete.")
print("Best model:", best_model_name)
print("Best ROC-AUC:", best_auc)
print("Saved model to:", MODELS_DIR / "model.pkl")
print("Saved results to:", REPORTS_DIR / "model_results.csv")
