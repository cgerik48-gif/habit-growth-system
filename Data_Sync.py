import json
import gspread
import pandas as pd
from google.oauth2.service_account import Credentials


def sync_data():
    # 1. Load configuration from your config.json
    try:
        with open('config.json', 'r') as f:
            config = json.load(f)
    except FileNotFoundError:
        print("Error: config.json file not found.")
        return

    # 2. Setup Authentication
    scope = ["https://www.googleapis.com/auth/spreadsheets", "https://www.googleapis.com/auth/drive"]
    try:
        creds = Credentials.from_service_account_file('credentials.json', scopes=scope)
        client = gspread.authorize(creds)
    except Exception as e:
        print(f"Authentication failed: {e}")
        return

    # 3. Sync each tab defined in config
    for tab_name, filename in config["TARGETS"].items():
        try:
            print(f"Syncing {tab_name}...")
            sheet = client.open_by_key(config["SHEET_ID"]).worksheet(tab_name)
            data = sheet.get_all_records()

            # Convert to DataFrame
            df = pd.DataFrame(data)

            # Cleanup: Remove rows that are completely empty
            df = df.dropna(how='all')

            # Only save if there is actual data (skips headers-only sheets)
            if df.empty:
                print(f"Note: {tab_name} is empty, skipping file creation.")
                continue

            df.to_csv(filename, index=False)
            print(f"Successfully saved {filename}")

        except Exception as e:
            print(f"Error syncing {tab_name}: {e}")


if __name__ == "__main__":
    sync_data()