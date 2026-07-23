from Data_Sync import sync_data
from Predictive_Engine import build_predictive_engine
from Risk_Calculator import analyze_risk

def main():
    print("\n--- STEP 0: Auto-Syncing Google Sheets Data ---")
    sync_data()
    print("\n--- Sync Complete. Proceeding to Engine ---")

    print("\n--- STEP 1: Building Predictive Engine ---")
    build_predictive_engine()
    print("\n--- Engine Complete. Proceeding to Risk Calculator ---")

    print("\n--- STEP 2: Analyzing Risk ---")
    analyze_risk()
    print("\n--- Risk Analysis Complete. ---")

if __name__ == "__main__":
    main()