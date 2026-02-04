import streamlit as st
import joblib
import os
import numpy as np
import matplotlib.pyplot as plt

# --- 1. PAGE CONFIGURATION ---
st.set_page_config(
    page_title="HealthAI | Early Risk Predictor",
    page_icon="⚕️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# --- 2. PATH RESOLUTION & ASSET LOADING ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(os.path.dirname(BASE_DIR), "model")

def load_css():
    css_path = os.path.join(BASE_DIR, "styles.css")
    if os.path.exists(css_path):
        with open(css_path) as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

@st.cache_resource
def load_ml_assets():
    try:
        heart_model = joblib.load(os.path.join(MODEL_DIR, "heart_model.joblib"))
        heart_scaler = joblib.load(os.path.join(MODEL_DIR, "heart_scaler.joblib"))
        diabetes_model = joblib.load(os.path.join(MODEL_DIR, "diabetes_model.joblib"))
        diabetes_scaler = joblib.load(os.path.join(MODEL_DIR, "diabetes_scaler.joblib"))
        return heart_model, heart_scaler, diabetes_model, diabetes_scaler
    except Exception as e:
        st.error(f"Error loading models: {e}")
        return None, None, None, None

load_css()
h_model, h_scaler, d_model, d_scaler = load_ml_assets()

# --- 3. HELPER FUNCTIONS ---
def get_risk_status(prob):
    prob_pct = prob * 100
    if prob_pct >= 70:
        return "High Risk", "risk-high", prob_pct
    elif 40 <= prob_pct < 70:
        return "Medium Risk", "risk-medium", prob_pct
    else:
        return "Low Risk", "risk-low", prob_pct

# --- 4. UI HEADER ---
st.markdown("""
    <div class='main-card'>
        <h1>⚕️ AI Early Disease Risk Predictor</h1>
        <p style='color: #333; opacity: 0.8; font-size: 1.1rem;'>Precise Medical Analysis & Risk Stratification</p>
    </div>
""", unsafe_allow_html=True)

# --- 5. MAIN LAYOUT ---
col_input, col_result = st.columns([1, 1.2], gap="large")

with col_input:
    st.markdown("<div class='card'><h3>📋 Patient Metrics</h3>", unsafe_allow_html=True)
    
    with st.form("risk_form"):
        # Demographics
        c_dem1, c_dem2 = st.columns(2)
        age = c_dem1.number_input("Age (Years)", min_value=1, max_value=120, value=45)
        gender = c_dem2.selectbox("Gender", ["Male", "Female", "Other"])
        
        # Clinical Values
        c1, c2 = st.columns(2)
        bmi = c1.number_input("BMI (kg/m²)", min_value=10.0, max_value=60.0, value=25.5, step=0.1)
        glucose = c2.number_input("Glucose (mg/dL)", min_value=50, max_value=400, value=100)
        
        c3, c4 = st.columns(2)
        sys_bp = c3.number_input("Systolic BP (mmHg)", min_value=80, max_value=250, value=120)
        dia_bp = c4.number_input("Diastolic BP (mmHg)", min_value=40, max_value=150, value=80)
        
        chol = st.number_input("Cholesterol (mg/dL)", min_value=100, max_value=500, value=190)
        
        # Lifestyle Factors
        c5, c6 = st.columns(2)
        smoking = c5.selectbox("Smoking Status", ["Never", "Former", "Current"])
        alcohol = c6.selectbox("Alcohol Intake", ["None", "Occasional", "Frequent"])
        
        submit = st.form_submit_button("ANALYZE RISK PROFILE")
    st.markdown("</div>", unsafe_allow_html=True)

with col_result:
    if submit:
        # --- MODEL INFERENCE ---
        # Heart Input: [age, trestbps, chol, thalach, oldpeak]
        heart_features = np.array([[age, sys_bp, chol, 150, 1.0]])
        heart_scaled = h_scaler.transform(heart_features)
        heart_prob = h_model.predict_proba(heart_scaled)[0][1]
        h_label, h_class, h_pct = get_risk_status(heart_prob)

        # Diabetes Input: [age, glucose, bmi]
        diab_features = np.array([[age, glucose, bmi]])
        diab_scaled = d_scaler.transform(diab_features)
        diab_prob = d_model.predict_proba(diab_scaled)[0][1]
        d_label, d_class, d_pct = get_risk_status(diab_prob)

        # --- UI RESULTS DISPLAY ---
        st.markdown("<div class='card-container'>", unsafe_allow_html=True)
        
        # Result Cards
        res_c1, res_c2 = st.columns(2)
        with res_c1:
            st.markdown(f"""
                <div class='card'>
                    <h4 style='text-align:center; text-transform: uppercase; font-size: 0.8rem; opacity: 0.7;'>Heart Disease</h4>
                    <div class='{h_class}'>{h_label}</div>
                    <h2 style='text-align:center;'>{h_pct:.1f}%</h2>
                </div>
            """, unsafe_allow_html=True)
        
        with res_c2:
            st.markdown(f"""
                <div class='card'>
                    <h4 style='text-align:center; text-transform: uppercase; font-size: 0.8rem; opacity: 0.7;'>Diabetes Risk</h4>
                    <div class='{d_class}'>{d_label}</div>
                    <h2 style='text-align:center;'>{d_pct:.1f}%</h2>
                </div>
            """, unsafe_allow_html=True)
        
        # Visual Analytics
        st.markdown("<div class='card'><h4>Health Metrics Overview</h4>", unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(6, 3.2), facecolor='white')
        metrics = ['BMI', 'Glucose', 'Cholesterol', 'Sys BP']
        values = [bmi, glucose/2, chol/2, sys_bp] # Normalized for visual comparison
        
        # Using the defined UI palette for the chart
        colors = ['#1E90FF', '#2ECC71', '#F1C40F', '#E74C3C']
        
        bars = ax.barh(metrics, values, color=colors, alpha=0.9)
        ax.set_title("Clinical Values Comparison (Normalized)", color='#0A3D62', fontsize=10)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.tick_params(axis='both', which='major', labelsize=8)
        
        st.pyplot(fig)
        st.markdown("</div>", unsafe_allow_html=True)
        
        st.markdown("</div>", unsafe_allow_html=True)
    else:
        st.markdown("""
            <div style='height: 100%; display: flex; align-items: center; justify-content: center; border: 2px dashed #E0E6ED; border-radius: 12px; padding: 100px;'>
                <div style='text-align: center;'>
                    <p style='color: #0A3D62; font-size: 4rem; margin: 0;'>🩺</p>
                    <p style='color: #64748B; font-weight: 500;'>Patient Analysis Pending</p>
                    <p style='color: #94A3B8; font-size: 0.9rem;'>Complete the clinical form and click 'Analyze' to view results.</p>
                </div>
            </div>
        """, unsafe_allow_html=True)

# --- 6. FOOTER ---
st.markdown("<br>", unsafe_allow_html=True)
st.markdown("""
    <div class='health-tip'>
        <strong>⚠️ Medical Disclaimer:</strong> This AI predictor is for educational and screening assistance purposes only. 
        It is not a diagnostic tool. Clinical decisions must be made by qualified medical professionals.
    </div>
""", unsafe_allow_html=True)

st.markdown("""
    <div style='text-align: center; color: #94A3B8; font-size: 0.8rem; margin-top: 30px; padding-bottom: 20px;'>
        Built for Hackathon Excellence | Healthcare AI Lab 2026
    </div>
""", unsafe_allow_html=True)