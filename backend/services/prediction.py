import joblib
from pathlib import Path

ROOT = Path(__file__).parent.parent.parent

# Charge le modèle et le scaler au démarrage
model  = joblib.load(ROOT / "models" / "best_model.pkl")
scaler = joblib.load(ROOT / "models" / "scaler.pkl")

def predict(data_sample):

    data_sample[['Amount', 'Time']] = scaler.transform(data_sample[['Amount', 'Time']])
    # Prédiction
    probability = model.predict_proba(data_sample)[0][1]
    is_fraud    = bool(probability > 0.5)

    # Niveau de risque
    if probability < 0.3:
        risk_level = "low"
    elif probability < 0.7:
        risk_level = "medium"
    else:
        risk_level = "high"

    return probability , is_fraud , risk_level