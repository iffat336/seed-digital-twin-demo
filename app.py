import streamlit as st
import joblib
import numpy as np
import pandas as pd
import time
import plotly.graph_objects as go

# --- PAGE CONFIG ---
st.set_page_config(
    page_title="Hygrothermal Digital Twin: Seed Mechanics",
    page_icon="🔬",
    layout="wide"
)

# --- CUSTOM CSS FOR PREMIUM LOOK ---
st.markdown("""
    <style>
    .main {
        background-color: #f8f9fa;
    }
    .stMetric {
        background-color: white;
        padding: 15px;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    </style>
    """, unsafe_allow_html=True)

# --- LOAD MODELS ---
@st.cache_resource
def load_assets():
    try:
        model = joblib.load('surrogate_model.pkl')
        scaler = joblib.load('scaler.pkl')
        return model, scaler
    except FileNotFoundError:
        return None, None

model, scaler = load_assets()

# --- SIDEBAR: RESEARCH CONTROL ---
st.sidebar.title("🔬 Research Bridge")
st.sidebar.markdown("""
**Focus**: Surrogate Modeling for Hygrothermal & Mechanical Characterization.
""")

st.sidebar.divider()
st.sidebar.subheader("Hygrothermal Variables")
rh = st.sidebar.slider("Relative Humidity (%)", 30.0, 95.0, 60.0, help="Vapor diffusion context (Szymczak-Graczyk logic)")
temp = st.sidebar.slider("Ambient Temperature (°C)", 5.0, 45.0, 22.0, help="Thermal performance variable")

st.sidebar.subheader("Mechanical Loading")
load = st.sidebar.slider("Mechanical Stress (kPa)", 50, 500, 150, help="Structural load context (Garbowski logic)")

st.sidebar.divider()
st.sidebar.info("""
**Research synergy applied:**
- **Szymczak-Graczyk**: Moisture permeability & Diffusion logic.
- **Garbowski**: FEM structural homogenization & Surrogate modeling.
""")

# --- MAIN DASHBOARD ---
st.title("🌱 Digital Twin: Seed Hygrothermal & Mechanical Stability")
st.markdown("### Predicting Material Degradation via ANN Surrogate Models")

if model is None:
    st.error("Model not found. Please run the simulation and training scripts first.")
    st.stop()

# --- REAL-TIME CALCULATION ---
# Features: relative_humidity_pct, temperature_c, mechanical_loading_kpa
input_feats = np.array([[rh, temp, load]])
input_scaled = scaler.transform(input_feats)

start_t = time.time()
stability_pred = model.predict(input_scaled)[0]
latency = (time.time() - start_t) * 1000

# --- METRICS ROW ---
m1, m2, m3 = st.columns(3)
with m1:
    st.metric("Effective Stability", f"{stability_pred:.2%}")
with m2:
    status = "OPTIMAL" if stability_pred > 0.7 else "DEGRADED" if stability_pred > 0.4 else "CRITICAL"
    st.metric("Material Status", status)
with m3:
    st.metric("Surrogate Latency", f"{latency:.2f} ms")

st.divider()

col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("Hygrothermal Degradation Logic")
    st.write("""
    This model simulates the **anisotropic diffusion** of moisture into a biological seed coat. 
    By applying **Inverse Analysis**, we can determine the 'effective properties' of the seed structure 
    under complex environmental wetting-drying cycles.
    """)
    
    if stability_pred < 0.4:
        st.error("🚨 **Structural Failure Risk**: The coupling of high RH and Loading has exceeded the material's load-bearing capacity.")
    elif stability_pred < 0.7:
        st.warning("⚠️ **Vapor Diffusion Hazard**: Moisture ingress is reducing structural homogenization efficiency.")
    else:
        st.success("✅ **Stable Configuration**: Material characteristics are within the safe hygrothermal range.")

with col2:
    # Scientific Gauge
    fig = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = stability_pred * 100,
        domain = {'x': [0, 1], 'y': [0, 1]},
        gauge = {
            'axis': {'range': [0, 100], 'tickwidth': 1},
            'bar': {'color': "darkgreen" if stability_pred > 0.7 else "orange" if stability_pred > 0.4 else "darkred"},
            'steps': [
                {'range': [0, 40], 'color': '#ff9999'},
                {'range': [40, 70], 'color': '#ffffcc'},
                {'range': [70, 100], 'color': '#ccffcc'}],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 90}}))
    
    fig.update_layout(height=350, margin=dict(l=20, r=20, t=50, b=20))
    st.plotly_chart(fig, use_container_width=True)

# --- SCIENTIFIC FOOTER ---
st.divider()
st.subheader("Methodology: Shell-to-Beam Surrogate Mapping")
st.write("""
This dashboard takes the output of a multi-step numerical homogenization (FEM) 
and maps it to a fast Artificial Neural Network (ANN). This enables a 
**Real-Time Digital Twin** that bridges Construction 4.0 and Agriculture 4.0.
""")

st.caption("Developed for PhD Interview Preparation - Poznań University of Life Sciences")
