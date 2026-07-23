import os
import sys
import pandas as pd
import numpy as np
import joblib


def analyze_risk():
    # Anchor working directory for PyCharm execution
    try:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        os.chdir(script_dir)
    except NameError:
        pass

    if not os.path.exists('relapse_predictor.pkl'):
        print("[ERROR] Model not found. Run Predictive_Engine.py first.")
        return

    model = joblib.load('relapse_predictor.pkl')
    emergency = pd.read_csv('emergency_log.csv')
    emergency.columns = emergency.columns.str.strip()

    # Standardize column naming
    target_col = 'Action Taken' if 'Action Taken' in emergency.columns else 'Status'
    effort_col = [c for c in emergency.columns if 'Effort' in c]
    if effort_col:
        emergency = emergency.rename(columns={effort_col[0]: 'Effort_Required'})
    else:
        emergency['Effort_Required'] = 0.0

    emergency['Relapse_Status'] = emergency[target_col].apply(
        lambda x: 1 if str(x).strip().lower() == 'relapsed' else 0
    )
    emergency['Effort_Required'] = pd.to_numeric(emergency['Effort_Required'], errors='coerce').fillna(0.0)
    emergency['ATR'] = pd.to_numeric(emergency['ATR'], errors='coerce').fillna(
        5.0) if 'ATR' in emergency.columns else 5.0
    emergency['Timestamp'] = pd.to_datetime(emergency['Timestamp'])
    emergency = emergency.sort_values('Timestamp').reset_index(drop=True)

    # ==========================================================
    # 1. RECONSTRUCT ACTUAL STATEFUL MEMORY
    # ==========================================================
    time_indexed = emergency.set_index('Timestamp')
    drain_7d = time_indexed['Effort_Required'].rolling('7D').sum().iloc[-1]
    momentum_7d = time_indexed['Relapse_Status'].rolling('7D').sum().iloc[-1]

    # Calculate true active resistance streak
    current_streak = 0
    for status in reversed(emergency['Relapse_Status'].values):
        if status == 1:
            break
        current_streak += 1

    expected_features = model.feature_names_in_
    base_data = {feat: 0.0 for feat in expected_features}

    # Inject accurate historical reality into the baseline
    base_data['Effort_Required'] = emergency['Effort_Required'].tail(3).mean()
    if 'ATR' in base_data: base_data['ATR'] = emergency['ATR'].tail(3).mean()
    if 'Willpower_Drain_7D' in base_data: base_data['Willpower_Drain_7D'] = float(drain_7d)
    if 'Relapse_Momentum_7D' in base_data: base_data['Relapse_Momentum_7D'] = float(momentum_7d)
    if 'Resistance_Streak' in base_data: base_data['Resistance_Streak'] = float(current_streak)

    # ==========================================================
    # 2. EVALUATE 24-HOUR HOURLY CURVE & DUAL METRICS
    # ==========================================================
    results = []
    for h in range(24):
        input_data = base_data.copy()
        if 'Hour' in input_data: input_data['Hour'] = float(h)
        input_df = pd.DataFrame([input_data])[expected_features]
        prob = model.predict_proba(input_df)[0][1]
        results.append(prob)

    # Sort hourly risks descending to find your daily "Critical Gauntlets"
    sorted_risks = sorted(results, reverse=True)
    top_3_gauntlets = sorted_risks[:3]

    # Metric 1: Realistic Daily Risk (Compounding only across your top 3 high-risk windows)
    realistic_daily_risk = 1.0 - np.prod([(1.0 - p) for p in top_3_gauntlets])

    # Metric 2: Long-Term Goal Tracker (Compounding across all 24 hours)
    long_term_compounded_risk = 1.0 - np.prod([(1.0 - p) for p in results])

    peak_risk = max(results)
    peak_time = np.argmax(results)
    avg_risk = np.mean(results)

    print("\n" + "=" * 68)
    print("--- SYSTEM RISK ASSESSMENT (DUAL-TIER FORECAST) ---")
    print("=" * 68)
    print(f"Active Resistance Streak Logged  : {current_streak} events")
    print(f"Current 7-Day Willpower Drain    : {drain_7d:.1f} effort points")
    print("-" * 68)
    print(f"REALISTIC DAILY FORECAST (Top 3) : {realistic_daily_risk:.1%} (Immediate 24h survival)")
    print(f"LONG-TERM GOAL (24h Compound)    : {long_term_compounded_risk:.1%} (Crush below 50% over time)")
    print("-" * 68)
    print(f"Peak Danger Hour Spike           : {peak_risk:.1%} (at {peak_time:02d}:00)")
    print(f"Baseline Average Hourly Risk     : {avg_risk:.1%}")
    print("=" * 68)

    # ==========================================================
    # 3. UPCOMING HOUR CHECK WITH CONTEXT OVERRIDES
    # ==========================================================
    choice = input("\nCalculate high-precision risk for the upcoming hour? (y/n): ").strip().lower()
    if choice == 'y':
        current_h = pd.Timestamp.now().hour
        input_data = base_data.copy()
        if 'Hour' in input_data: input_data['Hour'] = float(current_h)

        print(f"\nRefining conditions for {current_h:02d}:00...")
        for ctx in ['Stressed', 'In bed', 'Alone/ bedroom', 'Bored', 'Tired']:
            if ctx in expected_features:
                ans = input(f" -> Is '{ctx}' currently active? (y/n, default n): ").strip().lower()
                input_data[ctx] = 1.0 if ans == 'y' else 0.0

        input_df = pd.DataFrame([input_data])[expected_features]
        live_prob = model.predict_proba(input_df)[0][1]
        print(f"\n[>>>] LIVE PROBABILITY FOR {current_h:02d}:00 : {live_prob:.1%}")

    # ==========================================================
    # 4. GRANULAR HYPOTHETICAL MODE
    # ==========================================================
    choice_hypo = input("\nRun a granular hypothetical stress test? (y/n): ").strip().lower()
    if choice_hypo == 'y':
        print("\n--- GRANULAR HYPOTHETICAL MODE ---")
        print("Type 'exit' at any prompt to abort.\n")
        hypo_data = base_data.copy()
        try:
            for feat in expected_features:
                val = input(f"Input value for '{feat}' (Default {base_data.get(feat, 0):.2f}): ").strip()
                if val.lower() == 'exit': return
                if val:
                    hypo_data[feat] = float(val)

            hypo_df = pd.DataFrame([hypo_data])[expected_features]
            prob = model.predict_proba(hypo_df)[0][1]
            print(f"\n[!] HYPOTHETICAL RISK SCORE: {prob:.1%}")
        except Exception as e:
            print(f"Error: {e}. Please enter numeric values.")


if __name__ == "__main__":
    analyze_risk()