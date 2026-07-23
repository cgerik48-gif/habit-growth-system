import streamlit as st
import pandas as pd
from Data_Sync import sync_data
from Predictive_Engine import build_predictive_engine
from Risk_Calculator import analyze_risk

# Import your core engine if you want direct button logging
# from engine import NeuroMomentumEngine

st.set_page_config(
    page_title="Neuro-Momentum Engine",
    page_icon="⚡",
    layout="centered"
)

st.title("⚡ Neuro-Momentum Dashboard")
st.markdown("### Operational Intelligence & Risk Control")

# 1. Auto-Sync Section
with st.expander("🔄 Sync Data from Google Sheets", expanded=False):
    if st.button("Run Sync"):
        with st.spinner("Syncing with Google Sheets..."):
            sync_data()
            st.success("Sync Complete!")

st.divider()

# 2. Predictive Engine & Risk Analysis Output
st.subheader("📊 Live Telemetry & Risk Assessment")

if st.button("Run Risk Analysis & Engine", type="primary", use_container_width=True):
    with st.spinner("Processing telemetry and predictive models..."):
        # Run your pipeline functions
        build_predictive_engine()
        risk_output = analyze_risk()  # Ensure your analyze_risk returns data or prints to screen

        st.success("Analysis Complete.")
        st.markdown("Check your terminal or Google Sheets for detailed outputs, or log your action below.")

st.divider()

# 3. Action Logging (The Core Interaction)
st.subheader("🎯 Daily Action Log")
col1, col2 = st.columns(2)

with col1:
    if st.button("🛡️ RESIST", use_container_width=True, type="secondary"):
        st.toast("Recorded: RESIST. Momentum secured.")
        # Hook your engine 'RESIST' logic here

with col2:
    if st.button("⚠️ RELAPSE", use_container_width=True):
        st.warning("Recorded: RELAPSE. Chaser protocol engaged.")
        # Hook your engine 'RELAPSE' logic here