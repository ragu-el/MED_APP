from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd

# Load models using relative paths
models = {
    "Linear Regression": joblib.load("models/linear_regression.pkl"),
    "Random Forest": joblib.load("models/random_forest.pkl"),
    "Decision Tree": joblib.load("models/decision_tree.pkl"),
    "Gradient Boosting": joblib.load("models/gradient_boosting.pkl")
}

app = FastAPI(title="Health Insurance Charges Prediction API")

# Input schema
class InsuranceInput(BaseModel):
    age: int
    sex: str
    bmi: float
    children: int
    smoker: str
    region: str
    model_choice: str = "Linear Regression"

@app.post("/predict")
def predict_charges(data: InsuranceInput):
    # Prepare DataFrame
    input_data = pd.DataFrame({
        "age": [data.age],
        "sex": [data.sex],
        "bmi": [data.bmi],
        "children": [data.children],
        "smoker": [data.smoker],
        "region": [data.region]
    })

    # One-hot encode
    input_data = pd.get_dummies(input_data, columns=["sex", "smoker", "region"], drop_first=True)

    # Align columns
    for col in models["Linear Regression"].feature_names_in_:
        if col not in input_data.columns:
            input_data[col] = 0
    input_data = input_data[models["Linear Regression"].feature_names_in_]

    # Predict
    model = models[data.model_choice]
    prediction = model.predict(input_data)[0]

    return {"predicted_charges": round(float(prediction), 2)}
