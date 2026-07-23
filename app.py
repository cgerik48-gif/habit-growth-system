import streamlit as st
import io
import contextlib
import pandas as pd
import os
import sys
import subprocess

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Growth & Risk Command Center",
    page_icon="⚡",
    layout="wide"
)

st.title("⚡ System Command Center")
st.markdown("Execute syncs, run predictive models, and calculate live risk metrics without touching the terminal.")


# --- HELPER: DIRECT TERMINAL EXECUTION ---
def run_terminal_script(filename_options):
    """Executes a script in a separate process exactly like running it in terminal."""
    for fname in filename_options:
        if os.path.exists(fname):
            result = subprocess.run(
                [sys.executable, fname],
                capture_output=True,
                text=True
            )
            return result.stdout, result.stderr, fname
    return None, "File not found in repository.", None


# --- MODULE IMPORT FOR TAB 3 (Interactive UI) ---
try:
    import Risk_Calculator
except ImportError:
    try:
        import risk_calculator as Risk_Calculator
    except ImportError:
        Risk_Calculator = None

# --- NAVIGATION TABS ---
tab_sync, tab_engine, tab_risk = st.tabs([
    "🔄 Data Sync",
    "🧠 Predictive Engine",
    "⚠️ Risk Calculator"
])

# ==========================================
# TAB 1: DATA SYNC
# ==========================================
with tab_sync:
    st.header("Google Sheets Data Synchronization")
    st.write("Pull the latest logs from Google Sheets into the local system pipeline.")

    if st.button("🚀 Run Data Sync", type="primary", use_container_width=True):
        with st.spinner("Synchronizing with Google Sheets via terminal execution..."):
            stdout, stderr, found_file = run_terminal_script(["Data_Sync.py", "data_sync.py"])

            if not found_file:
                st.error("Error: Data_Sync.py could not be found in the project directory.")
            else:
                if stdout:
                    st.text_area("Terminal Sync Logs:", value=stdout, height=180)
                if stderr:
                    st.error(f"Sync Errors / Traceback:\n{stderr}")

                if not stderr and not stdout:
                    st.warning("Script executed, but produced no terminal print output.")
                elif not stderr:
                    st.success("Sync Complete!")

                # Preview updated CSV data automatically
                col1, col2 = st.columns(2)
                with col1:
                    if os.path.exists("daily_health.csv"):
                        st.subheader("Latest Daily Health Data")
                        df_health = pd.read_csv("daily_health.csv")
                        st.dataframe(df_health.tail(5), use_container_width=True)
                    else:
                        st.info("daily_health.csv not generated yet.")
                with col2:
                    if os.path.exists("emergency_log.csv"):
                        st.subheader("Latest Emergency Logs")
                        df_emerg = pd.read_csv("emergency_log.csv")
                        st.dataframe(df_emerg.tail(5), use_container_width=True)
                    else:
                        st.info("emergency_log.csv not generated yet.")

# ==========================================
# TAB 2: PREDICTIVE ENGINE
# ==========================================
with tab_engine:
    st.header("Predictive Analytics Engine")
    st.write("Run the machine learning pipeline to generate models and view all metrics, weights, and predictions.")

    if st.button("🧠 Execute Predictive Engine", type="primary", use_container_width=True):
        with st.spinner("Executing ML pipeline in terminal process..."):
            stdout, stderr, found_file = run_terminal_script(["Predictive_Engine.py", "predictive_engine.py"])

            if not found_file:
                st.error("Error: Predictive_Engine.py could not be found.")
            else:
                st.subheader("Complete Engine Terminal Output:")
                if stdout:
                    st.code(stdout, language="text")
                else:
                    st.warning("The script executed without printing text to stdout.")

                if stderr:
                    st.error(f"Engine Execution Errors:\n{stderr}")

                # Confirm the model file was actually created on the cloud disk
                if os.path.exists("relapse_predictor.pkl"):
                    st.success("✅ Model successfully trained and saved to disk as `relapse_predictor.pkl`!")
                else:
                    st.warning(
                        "⚠️ The script finished, but `relapse_predictor.pkl` was not generated. Check the error traceback above.")

# ==========================================
# TAB 3: RISK CALCULATOR INTERFACE
# ==========================================
with tab_risk:
    st.header("Live Risk Analysis")
    st.write("Input current environment parameters to calculate immediate vulnerability metrics.")

    # --- WEB UI INPUTS ---
    st.subheader("Current Status Parameters")

    col_a, col_b = st.columns(2)
    with col_a:
        is_alone = st.selectbox("Are you currently alone?", ["No", "Yes"])
        location = st.selectbox("Current Location:",
                                ["Public / Living Room", "Bedroom", "Bathroom", "Outside / Gym", "Other"])
        stress_level = st.slider("Current Stress / Fatigue Level (1-10):", min_value=1, max_value=10, value=5)

    with col_b:
        urge_intensity = st.slider("Urge / Distraction Intensity (1-10):", min_value=1, max_value=10, value=1)
        searching_behavior = st.selectbox("Active searching of triggers or explicit terms?", ["No", "Yes"])
        time_of_day = st.selectbox("Time Window:", ["Morning/Daytime", "Evening (6pm - 10pm)", "Late Night (10pm+)"])

    st.markdown("---")

    if st.button("⚠️ Run Risk Analysis", type="primary", use_container_width=True):
        if Risk_Calculator is None:
            st.error("Error: Risk_Calculator module not found.")
        elif not os.path.exists("relapse_predictor.pkl"):
            st.error(
                "🛑 [ERROR] Model not found on cloud disk. You must go to the 'Predictive Engine' tab and run the engine first to generate `relapse_predictor.pkl`!")
        else:
            with st.spinner("Calculating risk matrix..."):
                buffer = io.StringIO()
                with contextlib.redirect_stdout(buffer):
                    try:
                        inputs = {
                            "alone": is_alone == "Yes",
                            "location": location,
                            "stress": stress_level,
                            "urge": urge_intensity,
                            "searching": searching_behavior == "Yes",
                            "time": time_of_day
                        }

                        if hasattr(Risk_Calculator, 'analyze_risk'):
                            Risk_Calculator.analyze_risk(inputs)
                        elif hasattr(Risk_Calculator, 'calculate_risk'):
                            Risk_Calculator.calculate_risk(inputs)
                        elif hasattr(Risk_Calculator, 'main'):
                            Risk_Calculator.main(inputs)
                        else:
                            st.error("Could not find analyze_risk() inside Risk_Calculator.py")
                    except Exception as e:
                        print(f"\nRisk Calculation Error: {e}")

                risk_output = buffer.getvalue()
                st.subheader("Risk Analysis Results:")
                if risk_output.strip():
                    st.code(risk_output, language="text")
                else:
                    st.warning("Calculation finished, but no output was printed.")