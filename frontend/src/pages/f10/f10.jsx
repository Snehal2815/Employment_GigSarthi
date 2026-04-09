import React, { useState } from "react";
import "./f10.css";

function Feature10() {
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(false);

  const runFeature = async () => {
    setLoading(true);
    try {
      const res = await fetch("http://localhost:5000/api/features/10");
      const result = await res.json();
      setData(result);
    } catch (err) {
      console.error(err);
      alert("Error running feature");
    }
    setLoading(false);
  };

  return (
    <div className="f10-container">

      {/* BEAUTIFIED TITLE */}
      <h1 className="main-title">
        AI-Powered Gig Worker Income Intelligence
      </h1>
      <p className="subtitle">
        Advanced XGBoost Regression Model analyzing income patterns & workforce dynamics
      </p>

      <button className="run-btn" onClick={runFeature}>
        {loading ? "Running Analysis..." : "Run Model"}
      </button>

      {data && data.status === "success" && (
        <div className="results">

          {/* METRICS */}
          <div className="metrics">
            <div className="card">
              <h3>RMSE</h3>
              <p>{data.rmse.toFixed(2)}</p>
            </div>

            <div className="card">
              <h3>MAE</h3>
              <p>{data.mae.toFixed(2)}</p>
            </div>

            <div className="card highlight">
              <h3>R² Score</h3>
              <p>{data.r2_score.toFixed(4)}</p>
            </div>
          </div>

          {/* GRAPHS (2x2 GRID) */}
          <div className="graphs">

            <div className="graph-box">
              <h4>Actual vs Predicted Income</h4>
              <img src={`http://localhost:5000/${data.images.actual_vs_pred}`} />
            </div>

            <div className="graph-box">
              <h4>Residual Analysis</h4>
              <img src={`http://localhost:5000/${data.images.residual}`} />
            </div>

            <div className="graph-box">
              <h4>Distribution Comparison</h4>
              <img src={`http://localhost:5000/${data.images.distribution}`} />
            </div>

            <div className="graph-box">
              <h4>Feature Importance</h4>
              <img src={`http://localhost:5000/${data.images.importance}`} />
            </div>

          </div>

          {/* 🧠 INSIGHTS SECTION */}
          <div className="insights">
            <h2>🔍 Model Highlights</h2>

            <ul>
              <li>
                High <b>R² score</b> indicates strong predictive capability of the model.
              </li>
              <li>
                <b>Actual vs Predicted</b> shows how closely model matches real-world earnings.
              </li>
              <li>
                Residual spread indicates that prediction errors are random: model is performing well.
              </li>
              <li>
                Feature importance reveals that <b>jobs completed & hours worked</b> heavily influence earnings.
              </li>
              <li>
                Distribution similarity suggests model generalizes well across workers.
              </li>
            </ul>
          </div>

        </div>
      )}
    </div>
  );
}

export default Feature10;