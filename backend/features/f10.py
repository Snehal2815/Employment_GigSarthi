import pandas as pd
import numpy as np
import json
import os
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from xgboost import XGBRegressor

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# go to backend/
BACKEND_DIR = os.path.abspath(os.path.join(BASE_DIR, ".."))

# outputs folder
OUTPUT_DIR = os.path.join(BACKEND_DIR, "outputs")

os.makedirs(OUTPUT_DIR, exist_ok=True)

try:
    # ---------------- LOAD DATA ----------------

    file_path = os.path.join(BASE_DIR, "..", "..", "data", "raw", "gig_dirty_data.csv")
    df = pd.read_csv(file_path)

    # ---------------- CLEANING ----------------
    df.replace(["NaN", "", "??"], np.nan, inplace=True)

    df["hours_worked"] = df["hours_worked"].replace({
        "ten": 10, "eight": 8, "zero": 0, "none": np.nan
    })

    for col in ["hours_worked", "earnings_per_day", "jobs_completed", "rating"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df[
        (df["hours_worked"] >= 1) &
        (df["jobs_completed"] >= 0) &
        (df["rating"] >= 1) & (df["rating"] <= 5)
    ]

    df.fillna(df.median(numeric_only=True), inplace=True)

    # ---------------- OUTLIER REMOVAL ----------------
    Q1 = df["earnings_per_day"].quantile(0.25)
    Q3 = df["earnings_per_day"].quantile(0.75)
    IQR = Q3 - Q1

    df = df[
        (df["earnings_per_day"] >= Q1 - 1.5 * IQR) &
        (df["earnings_per_day"] <= Q3 + 1.5 * IQR)
    ]

    # ---------------- FEATURE ENGINEERING ----------------
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["day"] = df["date"].dt.day
    df["month"] = df["date"].dt.month
    df["weekday"] = df["date"].dt.weekday

    df.drop(columns=["date", "worker_id"], inplace=True)

    df = pd.get_dummies(df, columns=["platform", "location"], drop_first=True)

    # ---------------- SPLIT ----------------
    X = df.drop(columns=["earnings_per_day"])
    y = df["earnings_per_day"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # ---------------- MODEL ----------------
    model = XGBRegressor(n_estimators=100, learning_rate=0.1)
    model.fit(X_train, y_train)

    # ---------------- PREDICTIONS ----------------
    y_pred = model.predict(X_test)

    # ---------------- METRICS ----------------
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)

    os.makedirs("outputs", exist_ok=True)

    # ---------------- ACTUAL VS PREDICTED ----------------
    plt.figure(figsize=(6,6))
    plt.scatter(y_test, y_pred, alpha=0.6)

    min_val = min(min(y_test), min(y_pred))
    max_val = max(max(y_test), max(y_pred))

    plt.plot([min_val, max_val], [min_val, max_val], 'r--')
    plt.xlabel("Actual")
    plt.ylabel("Predicted")
    plt.title("Actual vs Predicted")

    plt.tight_layout()
    actual_vs_pred_path = os.path.join(OUTPUT_DIR, "f10_actual_vs_pred.png")
    plt.savefig(actual_vs_pred_path)
    plt.close()

    # ---------------- RESIDUAL PLOT ----------------
    residuals = y_test - y_pred

    plt.figure(figsize=(6,6))
    plt.scatter(y_pred, residuals, alpha=0.6)
    plt.axhline(y=0)

    plt.xlabel("Predicted")
    plt.ylabel("Residuals")
    plt.title("Residual Plot")

    plt.tight_layout()
    residual_path = os.path.join(OUTPUT_DIR, "f10_residual.png")
    plt.savefig(residual_path)
    plt.close()

    # ---------------- DISTRIBUTION ----------------
    plt.figure(figsize=(6,4))
    plt.hist(y_test, bins=20, alpha=0.5, label="Actual")
    plt.hist(y_pred, bins=20, alpha=0.5, label="Predicted")

    plt.legend()
    plt.title("Distribution Comparison")

    plt.tight_layout()
    distribution_path = os.path.join(OUTPUT_DIR, "f10_distribution.png")
    plt.savefig(distribution_path)
    plt.close()

    # ---------------- FEATURE IMPORTANCE ----------------
    importance = model.feature_importances_

    plt.figure(figsize=(8,5))
    plt.barh(X.columns, importance)
    plt.title("Feature Importance")

    plt.tight_layout()
    importance_path = os.path.join(OUTPUT_DIR, "f10_importance.png")
    plt.savefig(importance_path)
    plt.close()

    # ---------------- OUTPUT ----------------
    result = {
    "status": "success",
    "rmse": float(rmse),
    "mae": float(mae),
    "r2_score": float(r2),
    "images": {
        "actual_vs_pred": "outputs/f10_actual_vs_pred.png",
        "residual": "outputs/f10_residual.png",
        "distribution": "outputs/f10_distribution.png",
        "importance": "outputs/f10_importance.png"
    }
}

    print(json.dumps(result))

except Exception as e:
    print(json.dumps({"status": "error", "message": str(e)}))