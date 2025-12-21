# Diabetes Risk Prediction System

An end-to-end machine learning system that predicts the risk of diabetes using clinical features from the PIMA Indians Diabetes dataset. The project covers data exploration, model training, and deployment as a production-ready REST API using FastAPI and Render.

## Features
- Exploratory Data Analysis (EDA)
- Data preprocessing and feature engineering
- Random Forest classification model
- Modular prediction pipeline
- Ready for API deployment

## Tech Stack
- Python 3.12
- Pandas, NumPy
- Scikit-learn
- Jupyter Notebook
- FastAPI, Uvicorn

## Project Structure
    Diabetes-predictor/
        ├── notebooks/
        ├── src/
        ├── data/
        └── models/   ### Created at runtime (not committed)


## Live Deployment

Base URL:
   ```bash
   https://diabetes-risk-predictor-gzmz.onrender.com
   ```
Swagger UI (API Docs):
   ```bash
   https://diabetes-risk-predictor-gzmz.onrender.com/docs
   ```
## How to Run the Project Locally

1. Clone the repository
   ```bash
   git clone https://github.com/your-username/diabetes-risk-predictor.git
   cd diabetes-risk-predictor


2. Create and activate a virtual environment
   ```bash
   python -m venv venv
   venv\Scripts\activate ###windows


3. Install the required dependencies
   ```bash
   pip install -r requirements.txt


4. Train the model
   ```bash
   python notebooks/train_model.py
   ```

   Execute the cells sequentially to perform EDA and model training.
   Save the trained model and scaler to the models/ directory

   

5. Run the FastAPI application
   ```bash
   uvicorn api.main:app --reload
   ```

6. Open the browser and navigate to:
   ```bash
   http://127.0.0.1:8000/docs
   ```
## Notes
-The Random Forest model is used without feature scaling, as tree-based models are invariant to feature magnitude.
-A probability threshold of 0.3 is used to classify high-risk cases.
-Logistic Regression was used only during experimentation for baseline comparison and is not part of the production pipeline.
-Probability values are approximate due to the ensemble-based nature of Random Forest models.