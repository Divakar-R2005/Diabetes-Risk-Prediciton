from fastapi import FastAPI
from pydantic import BaseModel
from src.predict import predict_diabetes

app = FastAPI(
    title="Diabetes Risk Prediction API",
    description="API for predicting diabetes risk using a trained Random Forest model",
    version="1.0"
)

class InputData(BaseModel):
    Pregnancies: int
    Glucose: int
    BloodPressure: int
    SkinThickness: int
    Insulin: int
    BMI: float
    DiabetesPedigreeFunction: float
    Age: int


@app.get("/")
def health_check():
    return {"status": "API is running"}

@app.post("/predict")
def predict(data: InputData):
    probability = predict_diabetes(data.dict())

    THRESHOLD = 0.3
    risk = "High Risk" if probability >= THRESHOLD else "Low Risk"

    return {
        "risk": risk,
        "probability": round(probability, 2)
    }


# @app.post("/predict")
# def predict(input_data: DiabetesInput):
#     features = [
#         input_data.Pregnancies,
#         input_data.Glucose,
#         input_data.BloodPressure,
#         input_data.SkinThickness,
#         input_data.Insulin,
#         input_data.BMI,
#         input_data.DiabetesPedigreeFunction,
#         input_data.Age
#     ]

#     result = predict_diabetes(features)
#     THRESHOLD = 0.3
#     risk_label = "High Risk" if result["probability"] >= THRESHOLD else "Low Risk"


#     return {
#         "risk": risk_label,
#         "probability": result["probability"]
#     }