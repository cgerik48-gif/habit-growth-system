import os
import json
import gspread
import pandas as pd
from google.oauth2.service_account import Credentials

# --- 1. ABSOLUTE WORKING DIRECTORY LOCK ---
# Guarantees the script reads config.json and saves CSVs in the exact same folder as app.py
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(ROOT_DIR)


def sync_data():
    print("[STEP 1] Starting Google Sheets Data Synchronization...")

    # --- 2. CREDENTIALS CHECK & CLOUD FALLBACK ---
    creds_file = 'credentials.json'
    if not os.path.exists(creds_file):
        print(f"⚠️ '{creds_file}' not found locally. Checking Streamlit Cloud Secrets...")
        try:
            import streamlit as st
            if "GOOGLE_CREDENTIALS_JSON" in st.secrets:
                with open(creds_file, "w") as f:
                    f.write(st.secrets["GOOGLE_CREDENTIALS_JSON"].strip())
                print("✅ Successfully generated credentials.json from Streamlit Secrets.")
            else:
                print(
                    "🛑 [ERROR] No credentials.json found on disk and no GOOGLE_CREDENTIALS_JSON in Streamlit Secrets!")
                return
        except ImportError:
            print("🛑 [ERROR] credentials.json is missing and Streamlit secrets are inaccessible.")
            return

    # --- 3. LOAD CONFIGURATION ---
    try:
        with open('config.json', 'r') as f:
            config = json.load(f)
        print("✅ Successfully loaded config.json.")
    except FileNotFoundError:
        print(f"🛑 [ERROR] config.json file not found in directory: {ROOT_DIR}")
        return
    except json.JSONDecodeError as e:
        print(f"🛑 [ERROR] config.json is not valid JSON: {e}")
        return

    # --- 4. GOOGLE SHEETS AUTHENTICATION ---
    scope = ["https://www.googleapis.com/auth/spreadsheets", "https://www.googleapis.com/auth/drive"]
    try:
        creds = Credentials.from_service_account_file(creds_file, scopes=scope)
        client = gspread.authorize(creds)
        print("✅ Google Sheets API authentication successful.")
    except Exception as e:
        print(f"🛑 [ERROR] Authentication failed: {e}")
        return

    # --- 5. SYNC TARGETS & GUARANTEE CSV DELIVERY ---
    for tab_name, filename in config.get("TARGETS", {}).items():
        try:
            print(f"\n⏳ Syncing worksheet '{tab_name}' -> '{filename}'...")
            sheet = client.open_by_key(config["SHEET_ID"]).worksheet(tab_name)
            data = sheet.get_all_records()

            # --- THE FAULT-TOLERANT FIX: NEVER SKIP FILE CREATION ---
            # If get_all_records() returns empty (headers-only sheet), extract row 1 explicitly
            # so we still generate an empty CSV with valid column headers for downstream ML scripts.
            if not data:
                print(
                    f"⚠️ Note: '{tab_name}' has no data rows yet. Fetching schema headers to generate template CSV...")
                headers = sheet.row_values(1)

                # Fallback schema if row 1 is also completely blank in Google Sheets
                if not headers:
                    if "emergency" in filename.lower():
                        headers = ["Timestamp", "Location", "Alone", "Stress_Level", "Urge_Intensity",
                                   "Searching_Behavior", "Relapsed"]
                    elif "health" in filename.lower():
                        headers = ["Date", "Sleep_Hours", "Workout_Completed", "Calories", "Mood"]
                    else:
                        headers = ["Timestamp", "Value"]
                    print(f"⚠️ No headers found in row 1. Applied default schema: {headers}")

                df = pd.DataFrame(columns=headers)
            else:
                df = pd.DataFrame(data)
                # Cleanup: Remove rows that are completely empty across all columns
                df = df.dropna(how='all')

            # Save the CSV regardless of row count so downstream scripts NEVER throw FileNotFoundError
            df.to_csv(filename, index=False)
            print(f"✅ Successfully saved '{filename}' ({len(df)} rows, {len(df.columns)} columns)")

        except gspread.exceptions.WorksheetNotFound:
            print(f"🛑 [ERROR] Tab '{tab_name}' was not found in your Google Sheet! Check spelling in config.json.")
        except Exception as e:
            print(f"🛑 [ERROR] Failed syncing '{tab_name}': {e}")

    print("\n[SUCCESS] Data Sync pipeline execution complete.")


if __name__ == "__main__":
    sync_data()