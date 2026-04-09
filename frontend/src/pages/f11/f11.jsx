import React, { useState } from "react";
import "./f11.css";

function Feature11() {
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(false);

  const runFeature = async () => {
    setLoading(true);
    try {
      const res = await fetch("http://localhost:5000/api/features/11");
      const result = await res.json();
      setData(result);
    } catch (err) {
      console.error(err);
      alert("Error running feature");
    }
    setLoading(false);
  };

  return (
    <div className="f11-container">

      {/* 🔥 TITLE */}
      <h1 className="main-title">
        Model Optimization & Hyperparameter Intelligence
      </h1>
      <p className="subtitle">
        Step-by-step comparison, tuning, and selection of the best ML model architecture
      </p>

      <button className="run-btn" onClick={runFeature}>
        {loading ? "Running Optimization..." : "Run Optimization"}
      </button>

      {data && data.status === "success" && (
        <div className="results">

          {/* ================= STEP 1 ================= */}
          <div className="section">
            <h2>Step 1: Base Model Comparison</h2>

            <div className="metrics">
              {Object.entries(data.base_models).map(([name, val]) => (
                <div className="card" key={name}>
                  <h3>{name}</h3>
                  <p>R²: {val.r2.toFixed(3)}</p>
                  <p>RMSE: {val.rmse.toFixed(2)}</p>
                </div>
              ))}
            </div>

            <div className="graphs">
              <div className="graph-box full">
                <h4>Base Model Performance</h4>
                <img src={`http://localhost:5000/${data.graphs.base}`} />
              </div>
            </div>
          </div>

          {/* ================= STEP 2 ================= */}
          <div className="section">
            <h2>Step 2: Top Models Selected</h2>
            <p className="highlight-text">
              {data.top_models.join("  ➜  ")}
            </p>
          </div>

          {/* ================= STEP 3 ================= */}
          <div className="section">
            <h2>Step 3: Hyperparameter Tuning</h2>

            <div className="metrics">
              {Object.entries(data.tuned_models).map(([name, val]) => (
                <div className="card highlight" key={name}>
                  <h3>{name}</h3>
                  <p>R²: {val.r2.toFixed(3)}</p>
                  <p>RMSE: {val.rmse.toFixed(2)}</p>
                </div>
              ))}
            </div>

            <div className="graphs">
              <div className="graph-box">
                <h4>Tuned Model Performance</h4>
                <img src={`http://localhost:5000/${data.graphs.tuned}`} />
              </div>

              <div className="graph-box">
                <h4>Before vs After Optimization</h4>
                <img src={`http://localhost:5000/${data.graphs.compare}`} />
              </div>
            </div>
          </div>

          {/* ================= STEP 4 ================= */}
          <div className="section final">
            <h2>Final Best Model</h2>
            <h1>{data.best_model}</h1>
            <p>
              Selected based on highest R² score after hyperparameter tuning
              and cross-validation.
            </p>
          </div>

          {/* ================= INSIGHTS ================= */}
          <div className="insights">
            <h2>Optimization Insights</h2>
            <ul>
              <li>
                Initial model comparison helps identify strong baseline performers.
              </li>
              <li>
                Hyperparameter tuning significantly improves model accuracy.
              </li>
              <li>
                Ensemble models like <b>XGBoost / RandomForest</b> tend to dominate performance.
              </li>
              <li>
                Cross-validation ensures model generalization across unseen data.
              </li>
              <li>
                Final model selection is automated and data-driven.
              </li>
            </ul>
          </div>

        </div>
      )}
    </div>
  );
}

export default Feature11;