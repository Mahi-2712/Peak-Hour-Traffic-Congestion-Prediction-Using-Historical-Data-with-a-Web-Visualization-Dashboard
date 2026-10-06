import streamlit as st
import pandas as pd
import numpy as np
import folium
from streamlit_folium import st_folium
import time
import re
import pickle
import os

# --- PAGE CONFIGURATION & STYLING ---
st.set_page_config(
    page_title="Traffic Congestion Predictor",
    page_icon="🚦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom injection for a clean dark user interface layout framework
st.markdown("""
    <style>
    .main { background-color: #111827; color: #ffffff; }
    .stButton>button { width: 100%; background-color: #2563eb; color: white; border-radius: 6px; }
    .stButton>button:hover { background-color: #1d4ed8; color: white; }
    div[data-testid="stSidebarUserContent"] { background-color: #1f2937; }
    h1, h2, h3, p, span, label { color: #f3f4f6 !important; }
    input { background-color: #374151 !important; color: #ffffff !important; }
    </style>
""", unsafe_allow_html=True)

# Helper function to validate email text format patterns
def is_valid_email(email_str):
    pattern = r'^[\w\.-]+@[\w\.-]+\.\w+$'
    return re.match(pattern, email_str) is not None

# --- LOAD EXPORTED MACHINE LEARNING ARTIFACTS ---
@st.cache_resource
def load_ml_pipeline(model_path="traffic_model.pkl"):
    if os.path.exists(model_path):
        with open(model_path, 'rb') as f:
            return pickle.load(f)
    return None

ml_payload = load_ml_pipeline()

# --- LOGIN SESSION SYSTEM INITIALIZATION ---
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if "user_email" not in st.session_state:
    st.session_state.user_email = ""

# --- LOGIN SCREEN INTERFACE ---
if not st.session_state.logged_in:
    st.markdown("<h1 style='text-align: center; margin-top: 50px;'>🚦 College Project Portal</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; color: #9ca3af;'>Peak-Hour Traffic Congestion Prediction Dashboard</p>", unsafe_allow_html=True)
    
    _, col_login, _ = st.columns(3)
    
    with col_login:
        st.markdown("<div style='background-color: #1f2937; padding: 30px; border-radius: 12px; border: 1px solid #374151;'>", unsafe_allow_html=True)
        st.subheader("🔒 Account Sign In")
        
        email_input = st.text_input("Email Address", placeholder="name@example.com")
        password_input = st.text_input("Password", type="password", placeholder="Enter your secret password")
        
        if st.button("Secure Login"):
            if not email_input.strip():
                st.error("Email address field cannot be empty.")
            elif not is_valid_email(email_input):
                st.error("Please enter a valid email structure (e.g., user@domain.com).")
            elif not password_input:
                st.error("Password field cannot be empty.")
            else:
                if password_input == "project2026":
                    st.session_state.logged_in = True
                    st.session_state.user_email = email_input
                    st.success("Authentication successful! Entering visualization environment...")
                    time.sleep(0.5)
                    st.rerun()
                else:
                    st.error("Access denied. Incorrect password! Hint: Use password 'project2026'")
        st.markdown("</div>", unsafe_allow_html=True)
    st.stop()

# --- COMPONENT LAYOUT AFTER SUCCESSFUL LOGIN ---
st.sidebar.markdown(f"<h5>👤 User: {st.session_state.user_email}</h5>", unsafe_allow_html=True)
st.sidebar.markdown("<hr style='border-color: #374151;'>", unsafe_allow_html=True)

st.sidebar.header("🗺️ Portal Navigation")
page = st.sidebar.radio(
    "Select Interface Page:",
    ["🏠 Home & Live Spatial Map", "🔮 ML Congestion Forecasting", "📊 Historical Analytics"]
)

st.sidebar.markdown("<br><br><br>", unsafe_allow_html=True)
if st.sidebar.button("🔒 Terminate Session / Logout"):
    st.session_state.logged_in = False
    st.session_state.user_email = ""
    st.rerun()

# ====================================================================
# PAGE 1: HOME & LIVE SPATIAL MAP
# ====================================================================
if page == "🏠 Home & Live Spatial Map":
    st.title("📍 Live Traffic Overview & Geolocation Anchor")
    st.write("Visual regional traffic grid mapping modules.")
    
    col_input, col_map = st.columns(2)
    lat, lon = 17.5400, 78.4300  # Bhadurpalle, Hyderabad matrix coordinates fallback map
    
    with col_input:
        st.subheader("Grid Point Anchor")
        st.markdown("**Active Processing Center:**")
        st.code(f"Latitude:  {lat}\nLongitude: {lon}")
        st.metric("Live Network Nodes", "1,240", "+12%")
        st.metric("Congestion Factor Flow", "4.2", "-0.8")

    with col_map:
        m = folium.Map(location=[lat, lon], zoom_start=13)
        folium.Marker([lat, lon], popup="Core Traffic Matrix Node").add_to(m)
        st_folium(m, width="100%", height=450, key="live_traffic_canvas")

# ====================================================================
# PAGE 2: ML CONGESTION FORECASTING (LIVE PIPELINE INTEGRATED)
# ====================================================================
elif page == "🔮 ML Congestion Forecasting":
    st.title("🔮 Rush-Hour Predictive ML Pipeline")
    st.write("Forecast route saturation thresholds using your trained Scikit-Learn Random Forest Classifier model weights.")
    
    if ml_payload is None:
        st.warning("⚠️ Warning: Live 'traffic_model.pkl' not loaded. Please run 'python model_train.py' in your backend terminal to activate live forecasts!")
    else:
        col_feats, col_res = st.columns(2)
        with col_feats:
            st.subheader("Model Input Features Configuration")
            target_hour = st.slider("Target Prediction Hour (24h Clock)", 0, 23, 17)
            is_weekend = st.selectbox("Day Classification Matrix Type", ["Weekday (Mon-Fri)", "Weekend (Sat-Sun)"])
            is_holiday = st.checkbox("Is Selected Date an Official Public Holiday?")
            junction_count = st.number_input("Surrounding Junction Activity Weight", min_value=1, max_value=20, value=5)
            weather_sel = st.selectbox("Current Climate Conditions", ["Clear Skies", "Light Monsoon Drizzle", "Heavy Storm Outage"])
            vehicle_vol = st.slider("Estimated Local Vehicle Entry Volume Count", 100, 2000, 750)
            
            predict_trigger = st.button("Compute Forecast Execution")

        with col_res:
            st.subheader("Predictive ML Model Result Status")
            if predict_trigger:
                model = ml_payload["model_engine"]
                weather_enc = ml_payload["weather_mapper"]
                target_enc = ml_payload["target_mapper"]
                
                weekend_val = 1 if "Weekend" in is_weekend else 0
                holiday_val = 1 if is_holiday else 0
                
                # Fixed target matching indexing bug for unseen categorical transformations safely
                try:
                    weather_val = weather_enc.transform([weather_sel])[0]
                except ValueError:
                    weather_val = 0
                
                input_row = pd.DataFrame([[
                    target_hour, weekend_val, holiday_val, junction_count, weather_val, vehicle_vol
                ]], columns=ml_payload["features_list"])
                
                encoded_pred = model.predict(input_row)
                class_prediction = target_enc.inverse_transform(encoded_pred)[0]
                
                if class_prediction == "High":
                    st.error("### 🔴 LIVE CHIP ANALYSIS: HIGH CONGESTION PREDICTED")
                    st.write("Ensemble trees indicate severe flow bottleneck delays along target nodes. Rerouting rules advised.")
                elif class_prediction == "Medium":
                    st.warning("### 🟡 LIVE CHIP ANALYSIS: MEDIUM CONGESTION PREDICTED")
                    st.write("Moderate queue volumes accumulating across surrounding intersections.")
                else:
                    st.success("### 🟢 LIVE CHIP ANALYSIS: LOW/NORMAL SATURATION PREDICTED")
                    st.write("Flow metrics operating inside normal clearance limits. Trajectories stable.")
            else:
                st.info("Configure live traffic features in the left pane and execute model array predictions.")

# ====================================================================
# PAGE 3: HISTORICAL ANALYTICS
# ====================================================================
elif page == "📊 Historical Analytics":
    st.title("📊 Historical Data Processing Plots")
    st.write("Upload metric logging sheets to render analytical dataset summaries.")
    
    uploaded_file = st.file_uploader("Upload Historical Traffic CSV Log File:", type=["csv"])
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.success("Dataset successfully parsed!")
        st.dataframe(df.head(10), use_container_width=True)
    else:
        st.info("No logs uploaded yet. Showing baseline simulated network tracking curve metrics:")
        chart_data = pd.DataFrame(np.random.randn(24, 2), columns=['Main Highway Vector', 'Service Lane Grid'])
        st.line_chart(chart_data)
