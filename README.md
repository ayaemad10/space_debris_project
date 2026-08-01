# Space Debris Collision Prediction System

A modular ML system for predicting satellite-debris collision risk, built on
the ESA Kelvins Collision Avoidance Challenge feature set (`engineered_data_sample.csv`,
5,000 conjunction events x 103 engineered features).

## What was actually executed vs. what needs your machine

The authoring environment for this project has **no TensorFlow and no
network access**, so the CNN/RNN/LSTM training, SHAP, FastAPI and Streamlit
steps could not literally be run here. Everything below is marked accordingly:

| Step | Status |
|---|---|
| Data understanding, cleaning, EDA, feature-formula verification | ✅ Executed here, real output in `outputs/eda/` |
| GradientBoostingClassifier pipeline validation (proves the split/train/eval flow works) | ✅ Executed here — Accuracy 0.991, F1 0.953, ROC-AUC 0.999 on a held-out 20% split (see `outputs/baseline_validation_metrics.json`) |
| Feature importance (stand-in for SHAP) | ✅ Executed here, real output in `outputs/shap/feature_importance.png` |
| Decision engine | ✅ Executed here against real model predictions |
| CNN / RNN / LSTM training (`train_models.py`) | ⬜ Written, reviewed, **not run** — needs TensorFlow. Run it locally: `python src/scripts/train_models.py` |
| SHAP on the LSTM (`shap_explain.py`) | ⬜ Written, **not run** — needs `pip install shap` + TensorFlow |
| FastAPI backend / Streamlit dashboard | ⬜ Written, syntax-checked, **not run live** — needs TensorFlow, FastAPI, Streamlit installed |

Do not treat the "0.991 accuracy" number as the LSTM's — that number belongs
to the sklearn baseline used to verify the pipeline. Run `train_models.py`
locally to get real CNN/RNN/LSTM numbers.

## Two bugs fixed from the originally uploaded files

1. **`mission_id` doesn't exist.** The dataset has no identifier columns at
   all — every column is a scaled float feature. `app.py` and
   `mission_control.py` both called `df['mission_id']` and would have crashed
   on load. Replaced with `event_batch = row_index // 500`, a grouping over
   the *existing* rows (no synthetic data introduced).
2. **`risk` is not a 0-1 score.** It's `log10(collision probability)`,
   ranging roughly -30 to -3. The original decision logic
   (`risk_score > 0.8 -> CRITICAL`) would never fire on this data. The
   thresholds now apply to the *classifier's predicted probability*
   (0-1), and a binary target `risk_label = risk > -6` (Pc > 1e-6, the
   ESA Kelvins convention) was derived for training.

## Project Structure

```
space_debris_project/
├── data/
│   ├── engineered_data_sample.csv   # original upload (5000 x 103)
│   ├── cleaned_data.csv             # + risk_label column
│   └── engineered_data.csv          # final modeling table
├── models/
│   ├── lstm_model.keras             # original uploaded model
│   └── baseline_gbc.joblib          # real, trained sklearn baseline
├── src/
│   ├── backend/app.py               # FastAPI (bugs fixed)
│   ├── frontend/
│   │   ├── mission_control.py       # Streamlit (bugs fixed)
│   │   └── earth_simulation.html    # unmodified, works standalone
│   └── scripts/
│       ├── inspect_data.py
│       ├── preprocess.py
│       ├── eda.py
│       ├── feature_engineering.py
│       ├── train_models.py          # Keras CNN/RNN/LSTM — run locally
│       ├── train_baseline.py        # sklearn baseline — already run
│       ├── shap_explain.py          # SHAP on LSTM — run locally
│       ├── feature_importance_fallback.py  # already run
│       └── decision_engine.py
├── outputs/
│   ├── eda/                         # 4 real PNGs
│   ├── shap/                        # real feature-importance PNG/CSV
│   └── baseline_validation_metrics.json
├── Dockerfile.backend / Dockerfile.frontend / docker-compose.yml
└── requirements.txt
```

## Running it locally

```bash
pip install -r requirements.txt

python src/scripts/inspect_data.py
python src/scripts/preprocess.py
python src/scripts/eda.py
python src/scripts/feature_engineering.py
python src/scripts/train_models.py       # produces outputs/model_comparison.csv, saves best model
python src/scripts/shap_explain.py

python src/backend/app.py                 # http://localhost:8000/docs
streamlit run src/frontend/mission_control.py   # http://localhost:8501
```

### Docker

```bash
docker-compose up --build
# Backend:  http://localhost:8000
# Frontend: http://localhost:8501
```

## Known limitation: only a 5,000-row sample

`train_data.zip` (the full 162,634-row dataset mentioned in the original
spec) was not actually included in this upload — only
`engineered_data_sample.csv` (5,000 rows) was. All pipeline code works on
whatever CSV is at `data/engineered_data.csv`, so pointing it at the full
file (after running it through the same feature-engineering formulas) will
work without code changes — but the metrics above are only representative
of the 5,000-row sample.
