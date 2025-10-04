import streamlit as st
import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_error, r2_score

# Title
st.title("💰 Medical Insurance Cost Estimator")
st.write("Select a model and enter your details to estimate insurance charges.")

# Load dataset
@st.cache_data
def load_data():
    return pd.read_csv("C:/Users/Ragu/medical_insurance/medical_insurance.csv")

df = load_data()

# Features and target
X = df.drop("charges", axis=1)
y = df["charges"]

# Preprocessing
categorical = ["sex", "smoker", "region"]
numerical = ["age", "bmi", "children"]

preprocessor = ColumnTransformer([
    ("num", StandardScaler(), numerical),
    ("cat", OneHotEncoder(drop="first"), categorical)
])

# Model options
model_options = {
    "Linear Regression": LinearRegression(),
    "Ridge Regression": Ridge(),
    "Lasso Regression": Lasso(),
    "Decision Tree": DecisionTreeRegressor(),
    "Random Forest": RandomForestRegressor(),
    "Gradient Boosting": GradientBoostingRegressor(),
    "Support Vector Regressor": SVR()
}

# Model selector
selected_model_name = st.selectbox("Choose a regression model", list(model_options.keys()))
selected_model = model_options[selected_model_name]

# Build pipeline
pipeline = Pipeline([
    ("prep", preprocessor),
    ("model", selected_model)
])

# Train model
pipeline.fit(X, y)

# User input
st.subheader("📋 Your Information")
age = st.slider("Age", 18, 64, 30)
sex = st.selectbox("Gender", ["male", "female"])
bmi = st.number_input("BMI", min_value=10.0, max_value=60.0, value=25.0)
children = st.slider("Number of Children", 0, 5, 0)
smoker = st.selectbox("Smoker", ["yes", "no"])
region = st.selectbox("Region", ["northeast", "northwest", "southeast", "southwest"])

# Prediction
user_input = pd.DataFrame([{
    "age": age,
    "sex": sex,
    "bmi": bmi,
    "children": children,
    "smoker": smoker,
    "region": region
}])

if st.button("Estimate Cost"):
    prediction = pipeline.predict(user_input)
    st.success(f"Estimated Insurance Cost using {selected_model_name}: ₹{prediction[0]:,.2f}")


def get_model_scores(model):
    pipeline = Pipeline([
        ("prep", preprocessor),
        ("model", model)
    ])
    pipeline.fit(X, y)
    preds = pipeline.predict(X)
    return {
        "MAE": mean_absolute_error(y, preds),
        "R²": r2_score(y, preds)
    }

if st.checkbox("📈 Show model performance on full dataset"):
    st.subheader("Model Performance Comparison")
    for name, model in model_options.items():
        scores = get_model_scores(model)
        st.write(f"**{name}** → MAE: ₹{scores['MAE']:.2f}, R²: {scores['R²']:.4f}")

if hasattr(selected_model, "feature_importances_"):
    import matplotlib.pyplot as plt
    importances = selected_model.feature_importances_
    feature_names = preprocessor.get_feature_names_out()
    sorted_idx = np.argsort(importances)[::-1]

    st.subheader("🔍 Feature Importance")
    st.bar_chart(pd.Series(importances[sorted_idx], index=feature_names[sorted_idx]))