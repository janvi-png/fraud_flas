import os
import pickle
import sqlite3
from datetime import datetime

import pandas as pd
import requests
from flask import Flask, render_template, request


GITHUB_RAW = "https://raw.githubusercontent.com/janvi-png/fraud_flas/main/"
FILES = {
    "fraud_model.pkl": GITHUB_RAW + "fraud_model.pkl",
    "features.pkl":    GITHUB_RAW + "features.pkl",
    "encoders.pkl":    GITHUB_RAW + "encoders.pkl",
}

def download_once(url, fname):
    if os.path.exists(fname):
        return
    print(f"→ downloading {fname} …")
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    with open(fname, "wb") as f:
        f.write(r.content)

for f, u in FILES.items():
    download_once(u, f)

with open("fraud_model.pkl", "rb") as f:
    model = pickle.load(f)
with open("features.pkl", "rb") as f:
    FEATURES = pickle.load(f)
with open("encoders.pkl", "rb") as f:
    ENCODERS = pickle.load(f)

# build list-of-choices for every categorical feature
DROPDOWNS = {k: list(enc.classes_) for k, enc in ENCODERS.items()}

app = Flask(__name__)


def init_db():
    con = sqlite3.connect("history.db")
    con.execute(
        """
        CREATE TABLE IF NOT EXISTS history (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            ts          TEXT,
            payload     TEXT,
            prediction  INTEGER,
            probability REAL
        )
        """
    )
    con.close()

init_db()

def log_prediction(payload: dict, pred: int, prob: float) -> None:
    con = sqlite3.connect("history.db")
    con.execute(
        "INSERT INTO history (ts, payload, prediction, probability) VALUES (?, ?, ?, ?)",
        (datetime.now().isoformat(timespec="seconds"), str(payload), pred, prob),
    )
    con.commit()
    con.close()

@app.route("/", methods=["GET", "POST"])
def index():
    result, probability = None, None
    user_inputs = {}
    if request.method == "POST":
        try:
            values = []
            for feat in FEATURES:
                raw = request.form.get(feat)
                user_inputs[feat] = raw
                if feat in ENCODERS:
                    raw = ENCODERS[feat].transform([raw])[0]
                else:
                    raw = float(raw)
                values.append(raw)

            df = pd.DataFrame([values], columns=FEATURES)
            pred = int(model.predict(df)[0])
            prob = float(model.predict_proba(df)[0][pred])

            result = "Fraudulent" if pred == 1 else "Legitimate"
            probability = f"{prob*100:.2f}%"

            # NEW: persist the call
            log_prediction(user_inputs, pred, prob)
        except Exception as e:
            result, probability = f"❌ {e}", "N/A"

    return render_template(
        "index.html",
        result=result,
        probability=probability,
        inputs=user_inputs,
        features=FEATURES,
        dropdowns=DROPDOWNS,
    )

@app.route("/history")
def history():
    con = sqlite3.connect("history.db")
    rows = con.execute(
        "SELECT ts, payload, prediction, probability FROM history ORDER BY id DESC"
    ).fetchall()
    con.close()

    logs = [
        {
            "ts": r[0],
            "payload": r[1],
            "prediction": "Fraud" if r[2] == 1 else "Legit",
            "prob": f"{r[3]*100:.2f}%",
        }
        for r in rows
    ]
    return render_template("history.html", logs=logs)

if __name__ == "__main__":
    app.run(debug=True)
