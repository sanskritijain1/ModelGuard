from pathlib import Path
import sqlite3
import json

import pandas as pd
import streamlit as st
import plotly.express as px

BASE_DIR = Path(__file__).resolve().parents[1]
DB_PATH = BASE_DIR / "data" / "predictions.db"

st.set_page_config(
    page_title="ModelGuard Dashboard",
    page_icon="📊",
    layout="wide",
)

st.title("ModelGuard: ML Monitoring Dashboard")
st.write(
    "This dashboard monitors predictions made by the deployed Bank Marketing model."
)


def load_prediction_logs():
    if not DB_PATH.exists():
        return pd.DataFrame()

    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query("SELECT * FROM prediction_logs", conn)
    conn.close()

    if df.empty:
        return df

    df["timestamp"] = pd.to_datetime(df["timestamp"])

    parsed_inputs = df["input_data"].apply(json.loads)
    input_df = pd.json_normalize(parsed_inputs)

    df = pd.concat([df.drop(columns=["input_data"]), input_df], axis=1)

    return df


logs_df = load_prediction_logs()

if logs_df.empty:
    st.warning("No prediction logs found yet. Call the /predict endpoint first.")
    st.stop()

# Summary metrics
total_predictions = len(logs_df)
yes_predictions = (logs_df["prediction_label"] == "yes").sum()
no_predictions = (logs_df["prediction_label"] == "no").sum()
avg_probability = logs_df["probability_yes"].mean()
latest_model_version = logs_df["model_version"].iloc[-1]

col1, col2, col3, col4 = st.columns(4)

col1.metric("Total Predictions", total_predictions)
col2.metric("Yes Predictions", yes_predictions)
col3.metric("No Predictions", no_predictions)
col4.metric("Avg. Probability Yes", round(avg_probability, 4))

st.markdown("---")

st.subheader("Model Version")
st.info(f"Current logged model version: **{latest_model_version}**")

# Prediction distribution
st.subheader("Prediction Distribution")

prediction_counts = (
    logs_df["prediction_label"]
    .value_counts()
    .reset_index()
)

prediction_counts.columns = ["prediction_label", "count"]

fig_pred = px.bar(
    prediction_counts,
    x="prediction_label",
    y="count",
    title="Yes vs No Predictions",
)

st.plotly_chart(fig_pred, use_container_width=True)

# Probability trend
st.subheader("Probability Trend Over Time")

fig_prob = px.line(
    logs_df,
    x="timestamp",
    y="probability_yes",
    markers=True,
    title="Predicted Probability of Subscription Over Time",
)

st.plotly_chart(fig_prob, use_container_width=True)

# Probability distribution
st.subheader("Probability Distribution")

fig_hist = px.histogram(
    logs_df,
    x="probability_yes",
    nbins=20,
    title="Distribution of Predicted Probabilities",
)

st.plotly_chart(fig_hist, use_container_width=True)

# Logs table
st.subheader("Prediction Logs")

st.dataframe(
    logs_df.sort_values("timestamp", ascending=False),
    use_container_width=True,
)
