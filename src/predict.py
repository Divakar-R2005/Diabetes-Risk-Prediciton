import joblib
import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "diabetes_rf_model.pkl"

model = joblib.load(MODEL_PATH)

FEATURE_NAMES = [
    'Pregnancies',
    'Glucose',
    'BloodPressure',
    'SkinThickness',
    'Insulin',
    'BMI',
    'DiabetesPedigreeFunction',
    'Age'
]

def predict_diabetes(features):
    df_input = pd.DataFrame([features], columns=FEATURE_NAMES)
    prediction = model.predict(df_input)
    probability = model.predict_proba(df_input)[0][1]

    return {
        "prediction": int(prediction[0]),
        "probability": round(float(probability), 2)
    }
