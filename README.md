# ModelGuard: MLOps Model Monitoring Dashboard

ModelGuard is an end-to-end MLOps project for training, tracking, serving, and monitoring a machine learning model in a production-style workflow.

The current version includes:
- Bank Marketing dataset preprocessing
- Data leakage prevention by removing the `duration` column
- Time-based train/production split for monitoring simulation
- Model training with Logistic Regression and Random Forest
- MLflow experiment tracking
- Model comparison using Accuracy, Precision, Recall, F1-score, and ROC-AUC
- Best model selection and saving

## Dataset

This project uses the UCI Bank Marketing dataset.

Place the file below in:

```text
data/raw/bank-additional-full.csv
