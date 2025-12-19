# Diabetes Risk Prediction System

A machine learning project that predicts diabetes risk using clinical features from the PIMA Indians Diabetes dataset.The project includes exploratory data analysis, model training, and a FastAPI backend for real-time inference.

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
        └── models/

## Notes
- The API uses a probability threshold of 0.3 to classify high-risk cases.
- Probability outputs are approximate due to ensemble-based prediction.

## How to Run the Project

1. Clone the repository
   ```bash
   git clone https://github.com/your-username/diabetes-risk-predictor.git
   cd diabetes-risk-predictor


2. Create and activate a virtual environment
   ```bash
   python -m venv venv
   venv\Scripts\activate


3. Install the required dependencies
   ```bash
   pip install -r requirements.txt


4. Run exploratory data analysis and train the model

   Open the notebook in the notebooks/ directory:
   ```bash
   notebooks/EDA_and_Preprocessing.ipynb
   

Execute the cells sequentially to perform EDA and model training.
Save the trained model and scaler to the models/ directory

   

5. Run the FastAPI application
   ```bash
   uvicorn api.main:app --reload


Open the browser and navigate to:
http://127.0.0.1:8000/docs