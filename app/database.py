from pathlib import Path
import sqlite3
import json
from datetime import datetime

BASE_DIR = Path(__file__).resolve().parents[1]
DB_PATH = BASE_DIR / "data" / "predictions.db"


def init_db():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS prediction_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            input_data TEXT,
            prediction INTEGER,
            prediction_label TEXT,
            probability_yes REAL,
            model_version TEXT
        )
        """
    )

    conn.commit()
    conn.close()


def log_prediction(input_data, prediction, probability_yes, model_version):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    prediction_label = "yes" if int(prediction) == 1 else "no"

    cursor.execute(
        """
        INSERT INTO prediction_logs (
            timestamp,
            input_data,
            prediction,
            prediction_label,
            probability_yes,
            model_version
        )
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        (
            datetime.now().isoformat(timespec="seconds"),
            json.dumps(input_data),
            int(prediction),
            prediction_label,
            float(probability_yes),
            model_version,
        ),
    )

    conn.commit()
    conn.close()
