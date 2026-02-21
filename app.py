# --- DAY 3 ML SPRINT: STREAMLIT WEB APP (PROFESSIONAL UI) ---

import streamlit as st
import pandas as pd
import joblib

# 1. Page Configuration (Clean layout, no emojis)
st.set_page_config(page_title="Kenya Malnutrition AI", layout="centered")

# 2. Load the AI Model
@st.cache_resource 
def load_model():
    return joblib.load('stunting_predictor_model.pkl')

model = load_model()

# 3. Build the User Interface (Minimalist)
st.title("Climate-Driven Malnutrition Predictor")
st.markdown("""
**Model Architecture:** Random Forest Classifier (n_estimators=100)  
**Objective:** Predict the probability of chronic child stunting based on historical precipitation deficits and geographic location.
""")

st.divider()

# 4. The Interactive Sliders (The Inputs)
st.subheader("Simulate Climate Conditions")

col1, col2 = st.columns(2)

with col1:
    april_rain = st.slider("April 'Long Rains' (mm)", min_value=0, max_value=600, value=150)
    annual_rain = st.slider("Annual Rainfall (mm)", min_value=200, max_value=2500, value=1000)

with col2:
    lon = st.slider("Longitude (E)", min_value=34.0, max_value=42.0, value=35.0, format="%.2f")
    lat = st.slider("Latitude (N/S)", min_value=-5.0, max_value=5.0, value=1.0, format="%.2f")

# Map visualization
map_data = pd.DataFrame({'lat': [lat], 'lon': [lon]})
st.map(map_data, zoom=5)

# 5. The Prediction Engine
st.divider()
st.subheader("AI Risk Assessment")

input_data = pd.DataFrame({
    'april_rain_mm': [april_rain],
    'annual_rain': [annual_rain],
    'lon': [lon],
    'lat': [lat]
})

if st.button("Calculate Risk Probability", type="primary"):
    risk_probability = model.predict_proba(input_data)[0][1] * 100
    
    # Display the result with professional text formatting
    if risk_probability > 50:
        st.error(f"CRITICAL RISK: {risk_probability:.1f}% probability of stunting.")
        st.markdown("Threshold breached. Recommended intervention: Immediate nutritional aid and drought-resistant crop deployment.")
    elif risk_probability > 30:
        st.warning(f"MODERATE RISK: {risk_probability:.1f}% probability of stunting.")
    else:
        st.success(f"LOW RISK: {risk_probability:.1f}% probability of stunting.")