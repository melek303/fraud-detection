import pandas as pd
from fastapi import FastAPI, HTTPException
from backend.schemas.schemas import Transaction , PredictionResponse
from backend.services.prediction import predict

app = FastAPI(
    title       = "Fraud Detection API",
    description = "Détecte les transactions frauduleuses en temps réel",
    version     = "1.0.0"
)

@app.get("/")
def root():
    return {"Message" : "Fraud Detection Api is running!"}


@app.get("/health")
def health():
    return {"status" : "ok"}

@app.post("/predict", response_model=PredictionResponse)
def predict_route(transaction: Transaction):
    """
    Prédit si une transaction est frauduleuse.
    Retourne la probabilité et le niveau de risque.
    """
    try:
        # Convertit en DataFrame
        data = pd.DataFrame([transaction.dict()])
        
        probability , is_fraud , risk_level = predict(data)
        return PredictionResponse(
            is_fraud    = is_fraud,
            probability = round(float(probability), 4),
            risk_level  = risk_level
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))