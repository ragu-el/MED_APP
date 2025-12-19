import streamlit as st
import requests
import os

st.title("🏥 Health Insurance Charges Prediction")

# Use environment variable for API URL
API_URL = os.getenv("API_URL", "http://localhost:8000")

# Sidebar model choice
st.sidebar.header("Model Selection")
model_choice = st.sidebar.selectbox(
    "Choose a model",
    ["Linear Regression", "Random Forest", "Decision Tree", "Gradient Boosting"]
)

st.sidebar.info(f"API: {API_URL}")

# User inputs
st.header("Enter Your Information")

col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Age", min_value=18, max_value=100, value=30)
    sex = st.selectbox("Sex", ["male", "female"])
    bmi = st.number_input("BMI", min_value=10.0, max_value=60.0, value=25.0)

with col2:
    children = st.number_input("Children", min_value=0, max_value=5, value=0)
    smoker = st.selectbox("Smoker", ["yes", "no"])
    region = st.selectbox("Region", ["northeast", "northwest", "southeast", "southwest"])

# Call FastAPI
if st.button("🔮 Predict Charges", type="primary"):
    payload = {
        "age": age,
        "sex": sex,
        "bmi": bmi,
        "children": children,
        "smoker": smoker,
        "region": region,
        "model_choice": model_choice
    }
    
    with st.spinner("Calculating prediction..."):
        try:
            response = requests.post(f"{API_URL}/predict", json=payload, timeout=10)
            response.raise_for_status()
            result = response.json()
            
            st.success("✅ Prediction Complete!")
            st.metric(
                label="Predicted Insurance Charges",
                value=f"${result['predicted_charges']:,.2f}",
                delta=None
            )
            
            st.info(f"Model used: **{model_choice}**")
            
        except requests.exceptions.ConnectionError:
            st.error(f"❌ Could not connect to API at {API_URL}. Make sure the FastAPI backend is running.")
        except requests.exceptions.Timeout:
            st.error("⏱️ Request timed out. Please try again.")
        except requests.exceptions.HTTPError as e:
            st.error(f"❌ HTTP Error: {e}")
        except Exception as e:
            st.error(f"❌ Error: {str(e)}")
