import pandas as pd
import numpy as np
import json
import os
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor

try:
    # ================= LOAD =================
    file_path = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "..", "data", "raw", "gig_dirty_data.csv")
    )
    df = pd.read_csv(file_path)

    # ================= CLEANING =================
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

    # ================= OUTLIERS =================
    Q1 = df["earnings_per_day"].quantile(0.25)
    Q3 = df["earnings_per_day"].quantile(0.75)
    IQR = Q3 - Q1

    df = df[
        (df["earnings_per_day"] >= Q1 - 1.5 * IQR) &
        (df["earnings_per_day"] <= Q3 + 1.5 * IQR)
    ]

    # ================= FEATURE ENGINEERING =================
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["day"] = df["date"].dt.day
    df["month"] = df["date"].dt.month
    df["weekday"] = df["date"].dt.weekday

    df.drop(columns=["date", "worker_id"], inplace=True)

    df = pd.get_dummies(df, columns=["platform", "location"], drop_first=True)

    # ================= SPLIT =================
    X = df.drop(columns=["earnings_per_day"])
    y = df["earnings_per_day"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # ================= BASE MODELS =================
    models = {
        "LinearRegression": LinearRegression(),
        "RandomForest": RandomForestRegressor(),
        "GradientBoosting": GradientBoostingRegressor(),
        "XGBoost": XGBRegressor()
    }

    base_results = {}

    for name, model in models.items():
        model.fit(X_train, y_train)
        preds = model.predict(X_test)

        rmse = np.sqrt(mean_squared_error(y_test, preds))
        r2 = r2_score(y_test, preds)

        base_results[name] = {
            "rmse": float(rmse),
            "r2": float(r2)
        }

    # ================= SELECT TOP 2 =================
    sorted_models = sorted(base_results.items(), key=lambda x: x[1]["r2"], reverse=True)
    top_models = [sorted_models[0][0], sorted_models[1][0]]

    # ================= TUNING =================
    tuned_results = {}
    best_model_name = None
    best_r2 = -999

    param_grids = {
        "RandomForest": {
            "n_estimators": [100, 200],
            "max_depth": [5, 10, None]
        },
        "GradientBoosting": {
            "n_estimators": [100, 200],
            "learning_rate": [0.05, 0.1]
        },
        "XGBoost": {
            "n_estimators": [100, 200],
            "learning_rate": [0.05, 0.1],
            "max_depth": [3, 6]
        }
    }

    for name in top_models:
        model = models[name]
        grid = param_grids.get(name, {})

        if grid:
            gsearch = GridSearchCV(model, grid, cv=3, scoring="r2")
            gsearch.fit(X_train, y_train)

            best_model = gsearch.best_estimator_
            preds = best_model.predict(X_test)

            rmse = np.sqrt(mean_squared_error(y_test, preds))
            r2 = r2_score(y_test, preds)

            tuned_results[name] = {
                "rmse": float(rmse),
                "r2": float(r2),
                "best_params": gsearch.best_params_
            }

            if r2 > best_r2:
                best_r2 = r2
                best_model_name = name

    # ================= GRAPH SECTION =================
    os.makedirs("outputs", exist_ok=True)

    graphs = {}

    # 🔹 Base Model Comparison Graph
    names = list(base_results.keys())
    r2_scores = [base_results[n]["r2"] for n in names]

    plt.figure(figsize=(8,5))
    plt.bar(names, r2_scores)
    plt.title("Base Model Comparison (R² Score)")
    plt.xlabel("Models")
    plt.ylabel("R² Score")

    base_path = "outputs/f11_base.png"
    plt.savefig(base_path)
    plt.close()

    graphs["base"] = base_path


    # 🔹 Tuned Model Comparison Graph
    tuned_names = list(tuned_results.keys())
    tuned_scores = [tuned_results[n]["r2"] for n in tuned_names]

    plt.figure(figsize=(8,5))
    plt.bar(tuned_names, tuned_scores)
    plt.title("Tuned Models Performance (R² Score)")
    plt.xlabel("Models")
    plt.ylabel("R² Score")

    tuned_path = "outputs/f11_tuned.png"
    plt.savefig(tuned_path)
    plt.close()

    graphs["tuned"] = tuned_path


    # 🔹 Before vs After Graph
    before = [base_results[m]["r2"] for m in top_models]
    after = [tuned_results[m]["r2"] for m in top_models]

    x = np.arange(len(top_models))

    plt.figure(figsize=(8,5))
    plt.bar(x - 0.2, before, width=0.4, label="Before")
    plt.bar(x + 0.2, after, width=0.4, label="After")

    plt.xticks(x, top_models)
    plt.title("Before vs After Tuning")
    plt.ylabel("R² Score")
    plt.legend()

    compare_path = "outputs/f11_compare.png"
    plt.savefig(compare_path)
    plt.close()

    graphs["compare"] = compare_path

    # ================= OUTPUT =================
    result = {
        "status": "success",

        # 🔹 Step 1: Base model comparison
        "base_models": base_results,

        # 🔹 Step 2: Selected top models
        "top_models": top_models,

        # 🔹 Step 3: Tuned models + best params
        "tuned_models": tuned_results,

        # 🔹 Step 4: Final best model
        "best_model": best_model_name,

        # IMPORTANT: send ALL graphs
        "graphs": {
            "base": graphs["base"],
            "tuned": graphs["tuned"],
            "compare": graphs["compare"]
        }
    }



    print(json.dumps(result))

except Exception as e:
    print(json.dumps({
        "status": "error",
        "message": str(e)
    }))