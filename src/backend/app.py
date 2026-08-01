import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"   # hide TF C++ info/warnings
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"  # silence oneDNN floating-point notices

"""
FastAPI backend for the Space Debris Collision Prediction System.

Fixes applied in this version:
  1. Replaced deprecated @app.on_event("startup") with the modern
     asynccontextmanager `lifespan` pattern (FastAPI >= 0.93).
  2. Added robust error handling if the model or dataset file is missing
     at startup — the server now starts cleanly and returns a 503 on
     /predict instead of crashing the entire process.
  3. Added None guards on /statistics and /missions so they return a
     clear 503 rather than an AttributeError if called before startup
     completes or if loading failed.
  4. The dataset has NO `mission_id` column — replaced with a synthetic
     grouping over existing rows (event_batch = row_index // BATCH_SIZE).
  5. `risk` in this dataset is log10(collision probability), ranging
     roughly -30 to -3, NOT a 0-1 score. Decision thresholds are applied
     to the model's predicted probability output, not to the raw `risk`
     column (see src/scripts/decision_engine.py for the shared logic).
"""
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import numpy as np
import pandas as pd
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1] / "scripts"))
from decision_engine import decide  # noqa: E402

BASE_DIR = Path(__file__).resolve().parents[2]
MODEL_PATH = BASE_DIR / "models" / "lstm_model.keras"
DATA_PATH = BASE_DIR / "data" / "engineered_data.csv"
BATCH_SIZE = 500  # rows per synthetic "event_batch" grouping

model = None
df = None
startup_error: str = ""


# ---------------------------------------------------------------------------
# Lifespan: replaces the deprecated @app.on_event("startup")
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(application: FastAPI):
    """Load ML artefacts once at server start; release on shutdown."""
    global model, df, startup_error
    try:
        import tensorflow as tf  # imported lazily; env vars above silence the noise
        if not MODEL_PATH.exists():
            startup_error = f"Model file not found: {MODEL_PATH}"
        elif not DATA_PATH.exists():
            startup_error = f"Dataset file not found: {DATA_PATH}"
        else:
            model = tf.keras.models.load_model(MODEL_PATH)
            df = pd.read_csv(DATA_PATH)
            df["event_batch"] = df.index // BATCH_SIZE
            startup_error = ""
    except Exception as exc:  # noqa: BLE001
        startup_error = str(exc)

    yield  # server runs here

    # Cleanup on shutdown (nothing to clean for in-memory objects)
    model = None
    df = None


app = FastAPI(title="Space Debris Collision Prediction API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------
class PredictionRequest(BaseModel):
    features: list


class PredictionResponse(BaseModel):
    risk_probability: float
    risk_level: str
    recommended_action: str
    confidence: float


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------
@app.get("/")
def read_root():
    return {"message": "Space Debris Collision Prediction API"}


@app.get("/health")
def health_check():
    return {
        "status": "healthy" if not startup_error else "degraded",
        "model_loaded": model is not None,
        "startup_error": startup_error or None,
    }


@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest):
    if model is None:
        raise HTTPException(
            status_code=503,
            detail=startup_error or "Model is not loaded yet. Try again in a moment.",
        )
    try:
        features = np.array(request.features, dtype="float32").reshape(1, -1)
        reshaped = features.reshape((1, 1, features.shape[1]))
        probability = float(model.predict(reshaped, verbose=0)[0][0])

        decision = decide(probability)

        return PredictionResponse(
            risk_probability=probability,
            risk_level=decision.risk_level,
            recommended_action=decision.action,
            confidence=abs(probability - 0.5) * 2,
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/statistics")
def get_statistics():
    if df is None:
        raise HTTPException(
            status_code=503,
            detail=startup_error or "Data is not loaded yet.",
        )
    return {
        "total_events": len(df),
        "avg_log_risk": float(df["risk"].mean()),
        "avg_miss_distance": float(df["miss_distance"].mean()),
        "high_risk_events_pc_gt_1e-6": int((df["risk"] > -6).sum()),
    }


@app.get("/missions")
def get_missions():
    """Returns synthetic event-batch IDs since the dataset has no true
    mission identifier column (see module docstring for details)."""
    if df is None:
        raise HTTPException(
            status_code=503,
            detail=startup_error or "Data is not loaded yet.",
        )
    batches = sorted(df["event_batch"].unique().tolist())
    return {"event_batches": batches[:20]}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
