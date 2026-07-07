import os
import joblib
import pandas as pd

from fastapi import FastAPI, HTTPException
from api.schemas import TransactionInput


app = FastAPI(
    title= "Credit Card Fraud Detection API",
    description= "Production API for real-time bank transaction scoring",
    version='1.0.0'
)

SCALER_PATH = "models/robust_scaler.joblib"
MODEL_PATH = "models/logistic_regression_model.joblib"

if os.path.exists(SCALER_PATH) and os.path.exists(MODEL_PATH):
    print("Loging ML artifats into memory...")
    scaler = joblib.load(SCALER_PATH)
    model = joblib.load(MODEL_PATH)
    print("Artifacts loaded successfully! Api is ready")
else:
    print("Critical Error: Model or Scaler files not found in models/ directory!")
    scaler = None
    model = None

@app.get("/")
def read_root():
    """
    Root endpoint to check if the API is online.
    """
    return {
        "status":"online",
        "message": "Welcome to the Fraud Detection API. Endpoint is ready to process bank transactions!"
    }

@app.get("/predict")
@app.post("/predict", summary="Score a new transaction for fraud detection")
def predict_transcation(payload: TransactionInput):
    """
    Receives raw transaction data, scales it using the stored RobustScaler,
    and predicts whether it is a legitimate transaction or a fraud using the Logistic Regression model.
    """
    if model is None or scaler is None:
        raise HTTPException(
            status_code = 500,
            detail="Machine Learning models are not available on the server. Check logs."
        )
    
    try:
        # Step A: Convert the incoming Pydantic data into a Python dictionary
        input_data = payload.model_dump()

        # Step B: Convert it into a Pandas DataFrame
        df_input = pd.DataFrame([input_data])
        
        # Exact column order from your df.head() (Time, V1...V28, Amount)
        correct_columns_order = ['Time'] + [f'V{i}' for i in range(1, 29)] + ['Amount']
        df_input = df_input[correct_columns_order]
        
        # Step C: Scale the features using our trained RobustScaler
        X_scaled = scaler.transform(df_input.values)
        
        # Step D: Make the prediction (Passing raw scaled values)
        prediction = int(model.predict(X_scaled)[0])
        
        # Step E: Calculate the probability (confidence score) of being a fraud
        probabilities = model.predict_proba(X_scaled)
        fraud_probability = float(probabilities[0, 1]) # Probability of class 1 (Fraud)
        
        # Step F: Return the final response as a universal JSON object
        return {
            "is_fraud": True if prediction == 1 else False,
            "fraud_probability": round(fraud_probability, 4),
            "decision": "DENY" if prediction == 1 else "APPROVE"
        }
    
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        print(f"\n[ERRO REAL NO TERMINAL]:\n{error_details}")
        raise HTTPException(status_code=400, detail=f"Error type: {type(e).__name__} | Message: {str(e)}")
    
