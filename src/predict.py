from pathlib import Path
import joblib
import numpy as np

BASE_DIR = Path(__file__).resolve().parent.parent

model = joblib.load(BASE_DIR / "models" / "diabetes_rf_model.pkl")

FEATURE_ORDER = [
    'Pregnancies',
    'Glucose',
    'BloodPressure',
    'SkinThickness',
    'Insulin',
    'BMI',
    'DiabetesPedigreeFunction',
    'Age'
]

def predict_diabetes(input_dict):
    X = np.array(
        [input_dict[col] for col in FEATURE_ORDER],
        dtype=float
    ).reshape(1, -1)

    probability = model.predict_proba(X)[0][1]
    return probability
