import os
import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.preprocessing import MultiLabelBinarizer


def calculate_organic_s_curve_vectors(emergency_df):
    """
    Calculates both absolute Net_Momentum (M) and Momentum_Velocity (dM)
    using the state-dependent Gaussian-rate equations.

    Returns:
        momentum_list (list of floats): Absolute displacement positions.
        velocity_list (list of floats): Real-time rate of change values.
    """
    momentum_list = []
    velocity_list = []
    M = 0.0  # Start at neutral baseline

    # Gaussian parameters
    a_res, b_res, c_res, d_res = 2.5, 0.05, -2.0, 0.1
    u_rel, v_rel, w_rel, z_rel = 3.5, 0.04, 2.0, 0.2

    for _, row in emergency_df.iterrows():
        if row['Relapse_Status'] == 1:
            # Relapse pulls momentum down (Chaser effect peaks near M = 2.0)
            delta = u_rel * np.exp(-v_rel * (M - w_rel) ** 2) + z_rel
            M_new = M - delta
            velocity_list.append(-delta)
            M = M_new
        else:
            # Resist pushes momentum up (Breakthrough acceleration peaks near M = -2.0)
            delta = a_res * np.exp(-b_res * (M - c_res) ** 2) + d_res
            M_new = M + delta
            velocity_list.append(delta)
            M = M_new

        momentum_list.append(M)

    return momentum_list, velocity_list


def get_2d_diagnosis(M, dM):
    """
    Evaluates both position and velocity to determine
    the actual, real-time psychological state.
    """
    if M >= 5.0:
        if dM > 0:
            return "LEGENDARY SHIELD", "Unshakeable trajectory. Your habits are locked in and growing stronger."
        else:
            return "SHIELD EROSION", "CRITICAL WARNING: High safety margin, but actively decaying from a recent slip."

    elif 2.0 <= M < 5.0:
        if dM > 0:
            return "ACTIVE BUILD", "Developing stable momentum. Keep stacking days to lock in your shield."
        else:
            return "SLIPPERY SLOPE", "WARNING: Rapid loss of control. You are sliding toward the chaser cliff."

    elif -2.0 <= M < 2.0:
        if dM > 0:
            return "BREAKTHROUGH ACCELERATION", "High energy growth. You are actively surging out of danger."
        else:
            return "ACTIVE CRASH", "DANGER: Plunging down the cliff face. A high-priority intervention is needed."

    elif -6.0 <= M < -2.0:
        if dM > 0:
            return "GRITTY RECOVERY", "Grit phase. Initial resistance is hard, but you are building traction."
        else:
            return "COMPOUNDING CHAOS", "DANGER: Chaser effect is actively dragging you down. Break the cycle now."

    else:  # M < -6.0
        if dM > 0:
            return "SPARK OF LIFE", "Miracle action detected. You have initiated the climb out of a deep rut."
        else:
            return "STAGNANT DEPRESSION", "Severe dopaminergic deficit. Flat, bottomed-out habit rut."


def build_predictive_engine():
    print("[STEP 1] Initializing engine and resolving working directory...")

    try:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        os.chdir(script_dir)
        print(f"--> Working directory locked to: {script_dir}")
    except NameError:
        print(f"--> Using default working directory: {os.getcwd()}")

    log_file = 'emergency_log.csv'
    health_file = 'daily_health.csv'

    if not os.path.exists(log_file):
        print(f"\n[CRITICAL ERROR] '{log_file}' was not found in: {os.getcwd()}")
        print("-> Fix: Ensure 'emergency_log.csv' is saved in the exact same folder as this script.")
        return

    print("[STEP 2] Ingesting and cleaning emergency log...")
    emergency = pd.read_csv(log_file)
    emergency.columns = emergency.columns.str.strip()

    target_col = 'Action Taken' if 'Action Taken' in emergency.columns else 'Status'
    if target_col not in emergency.columns:
        print(f"\n[CRITICAL ERROR] Target column not found. Your CSV headers are: {list(emergency.columns)}")
        return

    emergency['Relapse_Status'] = emergency[target_col].apply(
        lambda x: 1 if str(x).strip().lower() == 'relapsed' else 0
    )

    # Standardize effort columns safely
    def get_col(df, name, default=0.0):
        if name in df.columns:
            return pd.to_numeric(df[name], errors='coerce').fillna(default)
        return pd.Series([default] * len(df))

    emergency['Effort_Required'] = get_col(emergency, 'Effort required (If Resisted)')
    if (emergency['Effort_Required'] == 0.0).all() and 'Effort' in emergency.columns:
        emergency['Effort_Required'] = get_col(emergency, 'Effort')

    emergency['Temptation Level'] = get_col(emergency, 'Temptation Level', default=5.0)
    emergency['ATR'] = get_col(emergency, 'ATR', default=5.0)

    print("[STEP 3] Generating stateful memory vectors (Gaussian-Rate Momentum & Velocity)...")
    emergency['Timestamp'] = pd.to_datetime(emergency['Timestamp'])
    emergency = emergency.sort_values('Timestamp').reset_index(drop=True)
    emergency['Hour'] = emergency['Timestamp'].dt.hour
    emergency['Date'] = emergency['Timestamp'].dt.date

    # Time-aware rolling window for willpower drain
    time_indexed = emergency.set_index('Timestamp')
    emergency['Willpower_Drain_7D'] = time_indexed['Effort_Required'].rolling('7D').sum().values

    # Calculate biological S-curve vectors (Absolute Position + Directional Velocity)
    momentum, velocity = calculate_organic_s_curve_vectors(emergency)
    emergency['Net_Momentum'] = momentum
    emergency['Momentum_Velocity'] = velocity

    # Merge daily health metrics
    if os.path.exists(health_file):
        print("[STEP 3b] Daily health file found. Merging metrics...")
        health = pd.read_csv(health_file)
        health.columns = health.columns.str.strip()
        if 'Timestamp' in health.columns:
            health['Date'] = pd.to_datetime(health['Timestamp']).dt.date
            health = health.drop(columns=['Timestamp'], errors='ignore')
        emergency = emergency.merge(health, on='Date', how='left')

    print("[STEP 4] Multi-label binarization on context tags...")
    ctx_series = emergency['Context'].fillna("").str.split(',').apply(lambda x: [i.strip() for i in x if i.strip()])
    mlb = MultiLabelBinarizer()
    ctx_encoded = pd.DataFrame(mlb.fit_transform(ctx_series), columns=mlb.classes_, index=emergency.index)

    # Drop old/redundant raw columns to prevent feature leakage
    meta_cols = [
        'Timestamp', 'Date', 'Context', target_col,
        'Relapse_Status', 'Outcome',
        'Effort required (If Resisted)', 'Effort'
    ]
    base_features = emergency.drop(columns=[c for c in meta_cols if c in emergency.columns], errors='ignore')

    features = pd.concat([base_features, ctx_encoded], axis=1)
    features = features.apply(pd.to_numeric, errors='coerce').fillna(0.0)
    target = emergency['Relapse_Status']

    print(f"[STEP 5] Training Random Forest on {len(features)} rows and {len(features.columns)} features...")
    rf_base = RandomForestClassifier(n_estimators=300, max_depth=10, class_weight='balanced', random_state=42)
    rf_base.fit(features, target)

    calibrated_model = CalibratedClassifierCV(estimator=rf_base, method='sigmoid', cv=3)
    calibrated_model.fit(features, target)

    joblib.dump(calibrated_model, 'relapse_predictor.pkl')
    joblib.dump(list(features.columns), 'model_features.pkl')
    print("[STEP 6] Models successfully serialized and saved to disk.")

    # ==========================================
    # OUTPUT 1: FULL FACTOR RESPONSIBILITY LIST
    # ==========================================
    print("\n" + "=" * 60)
    print("--- FACTOR RESPONSIBILITY RANKING (% CONTRIBUTION) ---")
    print("=" * 60)

    importances = rf_base.feature_importances_
    total_importance = np.sum(importances)
    percentage_contributions = (importances / total_importance) * 100.0

    factor_df = pd.DataFrame({
        'Factor': features.columns,
        'Contribution (%)': percentage_contributions
    }).sort_values('Contribution (%)', ascending=False).reset_index(drop=True)

    for idx, row in factor_df.iterrows():
        print(f"{idx + 1:02d}. {row['Factor'].ljust(30)} -> {row['Contribution (%)']:6.2f}%")

    # ==========================================
    # OUTPUT 2: HOURLY RISK IN A VACUUM
    # ==========================================
    print("\n" + "=" * 60)
    print("--- HOURLY RISK DANGER MAP (ISOLATED IN A VACUUM) ---")
    print("=" * 60)
    print("Evaluating hour-by-hour danger with all other factors held at safe/neutral baseline...\n")

    neutral_baseline = pd.DataFrame(0.0, index=[0], columns=features.columns)
    if 'Temptation Level' in neutral_baseline.columns: neutral_baseline['Temptation Level'] = 1.0
    if 'Effort_Required' in neutral_baseline.columns: neutral_baseline['Effort_Required'] = 0.0
    if 'Willpower_Drain_7D' in neutral_baseline.columns: neutral_baseline['Willpower_Drain_7D'] = 0.0
    if 'Net_Momentum' in neutral_baseline.columns: neutral_baseline[
        'Net_Momentum'] = 3.0  # Safe/active resilience baseline
    if 'Momentum_Velocity' in neutral_baseline.columns: neutral_baseline['Momentum_Velocity'] = 0.5
    if 'ATR' in neutral_baseline.columns: neutral_baseline['ATR'] = 5.0

    hourly_risk = {}
    for h in range(24):
        test_profile = neutral_baseline.copy()
        if 'Hour' in test_profile.columns:
            test_profile['Hour'] = float(h)
        risk_prob = calibrated_model.predict_proba(test_profile)[0][1]
        hourly_risk[h] = risk_prob

    sorted_hours = sorted(hourly_risk.items(), key=lambda x: x[1], reverse=True)

    for rank, (hour, risk) in enumerate(sorted_hours, 1):
        bar_len = int(risk * 40)
        bar = "█" * bar_len
        print(f"Rank {rank:02d} | Hour {hour:02d}:00 -> {risk:6.2%} | {bar}")

    # ==========================================
    # OUTPUT 3: REAL-TIME 2D MOMENTUM READOUT
    # ==========================================
    print("\n" + "=" * 60)
    print("--- LIVE 2D MOMENTUM STATUS REPORT ---")
    print("=" * 60)

    current_score = emergency['Net_Momentum'].iloc[-1]
    current_velocity = emergency['Momentum_Velocity'].iloc[-1]

    zone_title, zone_desc = get_2d_diagnosis(current_score, current_velocity)

    # Format trend symbol based on velocity direction
    if current_velocity > 1.5:
        trend_arrow = "▲▲▲ (Rapid Breakthrough Growth)"
    elif current_velocity > 0:
        trend_arrow = "▲   (Steady Upward Progress)"
    elif current_velocity < -2.0:
        trend_arrow = "▼▼▼ (Severe Accelerated Crash)"
    else:
        trend_arrow = "▼   (Downward Slip / Erosion)"

    print(f"Current Position (M)     : {current_score:+6.2f}")
    print(f"Current Velocity (dM)    : {current_velocity:+6.2f} {trend_arrow}")
    print(f"Diagnosed State Zone     : {zone_title}")
    print(f"Clinical Diagnosis       : {zone_desc}")
    print("-" * 60)
    print("Note: Positive velocity indicates strengthening neural pathways.")
    print("Negative velocity signals immediate vulnerability to behavioral loops.")
    print("=" * 60)

    print("\n[STEP 7] Pipeline execution complete.")


if __name__ == "__main__":
    build_predictive_engine()