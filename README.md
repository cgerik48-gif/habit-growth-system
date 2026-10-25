# Habit Growth System

A personal machine-learning pipeline that predicts when I'm most likely to slip on a habit I'm trying to break, and explains why. It turns my own self-logged data into a calibrated hourly risk forecast, a ranking of the factors that matter most, and a live risk check I can run from a web dashboard.

## Why I built it

Habit trackers count streaks, but they don't tell you *when* or *why* you're vulnerable. I wanted a system that learns my real triggers (time of day, stress, location, being alone, fatigue) from my own data and tells me ahead of time when to put guardrails up.

## How it works

```
Google Sheets (daily logs)
        |
        v
Data_Sync.py  ->  CSVs  ->  Predictive_Engine.py  ->  trained model (.pkl)
                                                      |
                                                      v
                                     Risk_Calculator.py  <-  Streamlit dashboard (app.py)
```

1. **Data sync** (`Data_Sync.py`): Pulls two logs from Google Sheets through a service account (gspread) and writes clean CSVs. Works locally with `credentials.json` or on Streamlit Cloud through secrets. Generates header-only templates when a sheet is empty so downstream steps never break.
2. **Predictive engine** (`Predictive_Engine.py`):
   - Builds time-aware features: hour of day, a 7-day rolling "effort drain", and multi-hot encoded context tags (MultiLabelBinarizer).
   - Models momentum with a custom state-dependent Gaussian-rate update, producing both a position (M) and a velocity (dM) for each event, so the system knows where I am *and* which way I'm heading.
   - Merges daily health metrics (sleep, workouts, mood) by date.
   - Trains a Random Forest (300 trees, balanced class weights) and calibrates it with Platt scaling (`CalibratedClassifierCV`) so outputs read as real probabilities.
   - Outputs a factor-importance ranking, an hour-by-hour risk map, and a 2D momentum diagnosis.
3. **Risk calculator** (`Risk_Calculator.py`):
   - Rebuilds my current state (streak, 7-day drain, recent effort) and scores all 24 hours.
   - Combines the three riskiest hours into a realistic daily forecast, plus a compounded 24-hour figure.
   - Scores the current hour live, with context overrides from the dashboard, and recommends an action at each risk tier.
4. **Dashboard** (`app.py`): A three-tab Streamlit app (Data Sync, Predictive Engine, Risk Calculator) so the whole pipeline runs without touching the terminal.

## Tech stack

Python, pandas, NumPy, scikit-learn, joblib, Streamlit, Google Sheets API (gspread, google-auth)

## Run it locally

```bash
git clone https://github.com/cgerik48-gif/habit-growth-system.git
cd habit-growth-system
pip install -r requirements.txt
```

1. Create a Google Cloud service account, share your sheet with it, and save its key as `credentials.json` in the project folder (it's git-ignored).
2. Set your sheet ID and tab names in `config.json`.
3. Run the full pipeline in the terminal:
   ```bash
   python Main.py
   ```
   or launch the dashboard:
   ```bash
   streamlit run app.py
   ```

## What I learned

- Designing features from messy, self-reported time-series data
- Why probability calibration matters when a model's output drives decisions
- Building an end-to-end pipeline: data ingestion, training, serialization, and a usable front end
- Handling secrets safely across local and cloud deployments

## Roadmap

- Hold-out evaluation (time-based train/test split) with ROC-AUC and Brier score reporting
- Charts for the hourly risk map and momentum curve in the dashboard
- Automated retraining on a schedule
- SHAP values for per-prediction explanations
