import streamlit as st
import io
import contextlib
import pandas as pd
import os

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Growth & Risk Command Center",
    page_icon="⚡",
    layout="wide"
)

st.title("⚡ System Command Center")
st.markdown("Execute syncs, run predictive models, and calculate live risk metrics without touching the terminal.")

# --- MODULE IMPORTS (Case-Safe) ---
try:
    import Data_Sync
except ImportError:
    try:
        import data_sync as Data_Sync
    except ImportError:
        Data_Sync = None

try:
    import Predictive_Engine
except ImportError:
    try:
        import predictive_engine as Predictive_Engine
    except ImportError:
        Predictive_Engine = None

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
        if Data_Sync is None:
            st.error("Error: Data_Sync module not found in repository.")
        else:
            with st.spinner("Synchronizing with Google Sheets..."):
                # Capture any terminal output during sync
                buffer = io.StringIO()
                with contextlib.redirect_stdout(buffer):
                    try:
                        # Call your sync function (adjust function name if different)
                        if hasattr(Data_Sync, 'sync_data'):
                            Data_Sync.sync_data()
                        elif hasattr(Data_Sync, 'main'):
                            Data_Sync.main()
                        else:
                            st.warning("Module found, but could not identify sync_data() or main() function.")
                    except Exception as e:
                        st.error(f"Sync failed with error: {e}")

                # Display terminal logs from sync
                logs = buffer.getvalue()
                if logs:
                    st.text_area("Terminal Sync Logs:", value=logs, height=150)

                st.success("Sync Complete!")

                # Preview the updated data automatically so you never have to open Sheets
                col1, col2 = st.columns(2)
                with col1:
                    if os.path.exists("daily_health.csv"):
                        st.subheader("Latest Daily Health Data")
                        df_health = pd.read_csv("daily_health.csv")
                        st.dataframe(df_health.tail(5), use_container_width=True)
                with col2:
                    if os.path.exists("emergency_log.csv"):
                        st.subheader("Latest Emergency Logs")
                        df_emerg = pd.read_csv("emergency_log.csv")
                        st.dataframe(df_emerg.tail(5), use_container_width=True)

# ==========================================
# TAB 2: PREDICTIVE ENGINE
# ==========================================
with tab_engine:
    st.header("Predictive Analytics Engine")
    st.write("Run the machine learning pipeline and view all metrics, weights, and predictions.")

    if st.button("🧠 Execute Predictive Engine", type="primary", use_container_width=True):
        if Predictive_Engine is None:
            st.error("Error: Predictive_Engine module not found in repository.")
        else:
            with st.spinner("Running machine learning pipeline..."):
                # Intercept terminal output
                buffer = io.StringIO()
                with contextlib.redirect_stdout(buffer):
                    try:
                        # Execute the engine's main loop or training function
                        if hasattr(Predictive_Engine, 'run_engine'):
                            Predictive_Engine.run_engine()
                        elif hasattr(Predictive_Engine, 'main'):
                            Predictive_Engine.main()
                        else:
                            # If script runs on import or lacks a main func, reload it
                            import importlib

                            importlib.reload(Predictive_Engine)
                    except Exception as e:
                        print(f"\nExecution Error: {e}")

                # Dump ALL terminal details directly to the screen
                engine_output = buffer.getvalue()
                st.subheader("Complete Engine Terminal Output:")
                if engine_output.strip():
                    st.code(engine_output, language="text")
                else:
                    st.warning(
                        "The engine ran, but did not print any text to the terminal. Add print() statements to your engine script to see metrics here.")

# ==========================================
# TAB 3: RISK CALCULATOR INTERFACE
# ==========================================
with tab_risk:
    st.header("Live Risk Analysis")
    st.write("Input current environment parameters to calculate immediate vulnerability metrics.")

    # --- WEB UI INPUTS (Replicating terminal input() questions) ---
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
        else:
            with st.spinner("Calculating risk matrix..."):
                buffer = io.StringIO()
                with contextlib.redirect_stdout(buffer):
                    try:
                        # Package UI inputs to pass to your calculator
                        inputs = {
                            "alone": is_alone == "Yes",
                            "location": location,
                            "stress": stress_level,
                            "urge": urge_intensity,
                            "searching": searching_behavior == "Yes",
                            "time": time_of_day
                        }

                        # Attempt to pass inputs if your calculator accepts arguments
                        if hasattr(Risk_Calculator, 'calculate_risk'):
                            Risk_Calculator.calculate_risk(inputs)
                        elif hasattr(Risk_Calculator, 'main'):
                            Risk_Calculator.main()
                        else:
                            # Fallback: reload module to trigger standalone execution
                            import importlib

                            importlib.reload(Risk_Calculator)
                    except Exception as e:
                        print(f"\nRisk Calculation Error: {e}")

                # Display complete terminal output
                risk_output = buffer.getvalue()
                st.subheader("Risk Analysis Results:")
                if risk_output.strip():
                    st.code(risk_output, language="text")
                else:
                    # If the calculator script doesn't print yet, show raw data calculation
                    st.write("Raw Input Matrix Captured:")
                    st.json(inputs)
                    st.info(
                        "To see your custom math here, ensure your Risk_Calculator.py prints its results to the console!")