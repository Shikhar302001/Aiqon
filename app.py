import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

# Page Configuration
st.set_page_config(
    page_title="Port Congestion & Delay Predictor",
    page_icon="🚢",
    layout="centered"
)

# Load Dataset & Train Model (Cached for performance)
@st.cache_resource
def load_and_train_model():
    df = pd.read_csv("port_congestion_synthetic_data.csv")
    df_model = pd.get_dummies(df, columns=["weather_condition", "port_name"], drop_first=True)
    
    X = df_model.drop(columns=["vessel_id", "arrival_delay_hrs", "congestion_level"])
    y = df_model["arrival_delay_hrs"]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    
    return rf, X

rf_model, X_columns = load_and_train_model()

# App UI Header
st.title("🚢 Port Congestion & Shipment Delay Prediction System")
st.markdown("Aiqon AI Internship - Take-Home Assignment Demo")
st.markdown("---")

# Sidebar / User Inputs
st.sidebar.header("📊 Operational Parameters")

vessel_arrival_count = st.sidebar.slider("Vessel Arrival Count", min_value=5, max_value=40, value=18, step=1)
port_capacity = st.sidebar.selectbox("Port Capacity", options=[10, 15, 20, 25], index=1)
queue_length = st.sidebar.slider("Queue Length (Vessels waiting)", min_value=0, max_value=20, value=4, step=1)
weather_condition = st.sidebar.selectbox("Weather Condition", options=["Good", "Moderate", "Bad", "Severe"])
historical_delay_hrs = st.sidebar.slider("Historical Delay Baseline (Hrs)", min_value=0.0, max_value=12.0, value=2.5, step=0.5)
port_name = st.sidebar.selectbox("Port Name", options=["Port of Singapore", "Port of Rotterdam", "Port of Los Angeles", "Port of Shanghai", "Port of Nhava Sheva"])

# Main Panel Display Current Input Summary
st.subheader("📋 Selected Scenario Overview")
col1, col2, col3 = st.columns(3)
col1.metric("Incoming Vessels", vessel_arrival_count)
col2.metric("Port Capacity", port_capacity)
col3.metric("Queue Length", queue_length)

st.write(f"**Port Location:** {port_name} | **Weather Status:** {weather_condition} | **Historical Baseline:** {historical_delay_hrs} hrs")

# Prediction Trigger Button
if st.button("🚀 Predict Delay & Congestion Risk", type="primary"):
    # Format Input Data
    input_data = {
        'vessel_arrival_count': [vessel_arrival_count],
        'port_capacity': [port_capacity],
        'queue_length': [queue_length],
        'historical_delay_hrs': [historical_delay_hrs]
    }
    
    for col in X_columns.columns:
        if col.startswith('weather_condition_') or col.startswith('port_name_'):
            input_data[col] = [1 if col == f"weather_condition_{weather_condition}" or col == f"port_name_{port_name}" else 0]
            
    input_df = pd.DataFrame(input_data)[X_columns.columns]
    
    # Predict delay using Random Forest
    pred_delay = rf_model.predict(input_df)[0]
    
    # Calculate Confidence Score using tree variance
    tree_preds = [tree.predict(input_df)[0] for tree in rf_model.estimators_]
    std_dev = np.std(tree_preds)
    confidence_score = max(0, min(100, round(100 - (std_dev * 10), 2)))
    
    # Generate Explanation Factors
    reasons = []
    if vessel_arrival_count > port_capacity:
        reasons.append(f"High vessel traffic ({vessel_arrival_count}) exceeding port capacity ({port_capacity})")
    if queue_length > 5:
        reasons.append(f"Long operational queue length of {queue_length} vessels")
    if weather_condition in ["Bad", "Severe"]:
        reasons.append(f"Adverse weather condition ({weather_condition}) affecting port maneuvers")
    if historical_delay_hrs > 5:
        reasons.append(f"High historical delay baseline recorded ({historical_delay_hrs} hrs)")
    
    if not reasons:
        reasons.append("Normal operational parameters with minimal congestion risk.")

    st.markdown("---")
    st.subheader("📈 Prediction Results")
    
    res_col1, res_col2 = st.columns(2)
    with res_col1:
        st.success(f"### Predicted Delay\n## **{round(pred_delay, 2)} Hours**")
    with res_col2:
        st.info(f"### Confidence Score\n## **{confidence_score}%**")
        
    st.markdown("### 🔍 Key Contributing Factors (Explainability):")
    for reason in reasons:
        st.markdown(f"- {reason}")