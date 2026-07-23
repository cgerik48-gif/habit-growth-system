import os
import sys
import pandas as pd
import numpy as np
import joblib


def analyze_risk(ui_inputs=None):
    # Anchor working directory for execution reliability
    try:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        os.chdir(script_dir)
    except NameError:
        pass

    if not os.path.exists('relapse_predictor.pkl'):
        print("[ERROR] Model not found. Run Predictive_Engine.py first.")
        return

    model = joblib.load('relapse_predictor.pkl')

    if not os.path.exists('emergency_log.csv'):
        print("[ERROR] emergency_log.csv not found. Run Data Sync first.")
        return

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
    drain_7d = time_indexed['Effort_Required'].rolling('7D').sum().iloc[-1] if not time_indexed.empty else 0.0
    momentum_7d = time_indexed['Relapse_Status'].rolling('7D').sum().iloc[-1] if not time_indexed.empty else 0.0

    # Calculate true active resistance streak
    current_streak = 0
    for status in reversed(emergency['Relapse_Status'].values):
        if status == 1:
            break
        current_streak += 1

    expected_features = model.feature_names_in_
    base_data = {feat: 0.0 for feat in expected_features}

    # Inject accurate historical reality into the baseline
    base_data['Effort_Required'] = emergency['Effort_Required'].tail(3).mean() if not emergency.empty else 0.0
    if 'ATR' in base_data: base_data['ATR'] = emergency['ATR'].tail(3).mean() if not emergency.empty else 5.0
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

    sorted_risks = sorted(results, reverse=True)
    top_3_gauntlets = sorted_risks[:3]

    realistic_daily_risk = 1.0 - np.prod([(1.0 - p) for p in top_3_gauntlets])
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
    # 3. LIVE UPCOMING HOUR & CONTEXT OVERRIDES (TIMEZONE AWARE)
    # ==========================================================
    try:
        current_h = pd.Timestamp.now(tz='America/New_York').hour
    except Exception:
        current_h = pd.Timestamp.now().hour

    live_data = base_data.copy()
    if 'Hour' in live_data: live_data['Hour'] = float(current_h)

    # PATH A: WEB UI MODE
    if ui_inputs is not None:
        print(f"\n--- LIVE HOURLY RISK FORECAST ({current_h:02d}:00 EST) ---")
        print("Applying Command Center UI Overrides:")

        active_triggers = []
        for feat in expected_features:
            feat_lower = feat.lower()
            if 'alone' in feat_lower and ui_inputs.get('alone'):
                live_data[feat] = 1.0
                active_triggers.append(feat)
            elif ('bed' in feat_lower or 'room' in feat_lower) and ui_inputs.get('location') == 'Bedroom':
                live_data[feat] = 1.0
                active_triggers.append(feat)
            elif 'stress' in feat_lower and ui_inputs.get('stress', 0) >= 6:
                live_data[feat] = 1.0
                active_triggers.append(f"{feat} (Level {ui_inputs.get('stress')})")
            elif 'urge' in feat_lower and ui_inputs.get('urge', 0) >= 5:
                live_data[feat] = float(ui_inputs.get('urge', 1))
                active_triggers.append(f"{feat} (Level {ui_inputs.get('urge')})")
            elif ('search' in feat_lower or 'freak' in feat_lower or 'trigger' in feat_lower) and ui_inputs.get(
                    'searching'):
                live_data[feat] = 1.0
                active_triggers.append(feat)

        if active_triggers:
            for trigger in active_triggers:
                print(f" [!] Active Factor: {trigger}")
        else:
            print(" [i] No high-risk environmental triggers active.")

        live_df = pd.DataFrame([live_data])[expected_features]
        live_prob = model.predict_proba(live_df)[0][1]

        print("-" * 68)
        print(f"[>>>] LIVE PROBABILITY FOR {current_h:02d}:00 : {live_prob:.1%}")
        print("-" * 68)

        if live_prob > 0.70 or ui_inputs.get('searching'):
            print("⚠️ ACTION REQUIRED: IMMEDIATE CIRCUIT BREAKER.")
            print("1. Terminate current digital session immediately.")
            print("2. Evacuate current room/isolation area and move to a public space.")
            print("3. Execute physical reset (maximum hang hold or 50 pushups).")
        elif live_prob > 0.40:
            print("⚡ WARNING: Elevated vulnerability window detected.")
            print("Maintain guardrail protocols. Avoid isolation.")
        else:
            print("✅ STATUS: Nominal. Continue scheduled execution.")
        print("=" * 68)
        return

    # PATH B: TERMINAL INTERACTIVE MODE
    try:
        choice = input(
            f"\nCalculate high-precision risk for upcoming hour ({current_h:02d}:00)? (y/n): ").strip().lower()
        if choice == 'y':
            print(f"\nRefining conditions for {current_h:02d}:00...")
            for ctx in ['Stressed', 'In bed', 'Alone/ bedroom', 'Bored', 'Tired']:
                if ctx in expected_features:
                    ans = input(f" -> Is '{ctx}' currently active? (y/n, default n): ").strip().lower()
                    live_data[ctx] = 1.0 if ans == 'y' else 0.0

            live_df = pd.DataFrame([live_data])[expected_features]
            live_prob = model.predict_proba(live_df)[0][1]
            print(f"\n[>>>] LIVE PROBABILITY FOR {current_h:02d}:00 : {live_prob:.1%}")

        choice_hypo = input("\nRun a granular hypothetical stress test? (y/n): ").strip().lower()
        if choice_hypo == 'y':
            print("\n--- GRANULAR HYPOTHETICAL MODE ---")
            print("Type 'exit' at any prompt to abort.\n")
            hypo_data = base_data.copy()
            for feat in expected_features:
                val = input(f"Input value for '{feat}' (Default {base_data.get(feat, 0):.2f}): ").strip()
                if val.lower() == 'exit': return
                if val:
                    hypo_data[feat] = float(val)

            hypo_df = pd.DataFrame([hypo_data])[expected_features]
            prob = model.predict_proba(hypo_df)[0][1]
            print(f"\n[!] HYPOTHETICAL RISK SCORE: {prob:.1%}")
    except (KeyboardInterrupt, EOFError):
        print("\n[i] Execution terminated.")


calculate_risk = analyze_risk
main = analyze_risk

if __name__ == "__main__":
    analyze_risk()