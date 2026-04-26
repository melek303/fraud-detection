import joblib
import mlflow
from pathlib import Path
from dotenv import load_dotenv

from preprocess import fetch_data, split_data, scale_features
from train import train_models, get_best_model
from tune import tune_model
from evaluate import evaluate

load_dotenv()

ROOT = Path(__file__).parent.parent.parent
tracking_path = str(ROOT / "mlruns").replace("\\", "/")

mlflow.set_tracking_uri(f"file:///{tracking_path}")
mlflow.set_experiment("fraud-detection")

query = f"""
    SELECT *
    FROM `fraud-detection-mlops-490823.fraud_detection.transactions`
"""

def save_artifacts(model, scaler):
    """Sauvegarde le modèle et le scaler"""
    models_dir = ROOT / "models"
    models_dir.mkdir(exist_ok=True)

    joblib.dump(model,  models_dir / "best_model.pkl")
    joblib.dump(scaler, models_dir / "scaler.pkl")

    print(f"\nModèle sauvegardé dans models/best_model.pkl ✅")
    print(f"Scaler sauvegardé dans models/scaler.pkl ✅")


def run_pipeline(n_trials: int = 50):
    """
    Pipeline complet :
    1. Preprocessing
    2. Compare plusieurs modèles
    3. Tune le meilleur
    4. Évalue sur test
    5. Sauvegarde
    """

    print("=" * 50)
    print("FRAUD DETECTION — TRAINING PIPELINE")
    print("=" * 50)

    # 1. Preprocessing
    print("\n[1/5] Preprocessing...")

    df = fetch_data(query)
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(df)

    from sklearn.preprocessing import RobustScaler
    scaler = RobustScaler()
    X_train, X_val, X_test, scaler = scale_features(
        X_train, X_val, X_test, scaler
    )
    print("Preprocessing terminé ✅")

    # 2. Compare les modèles
    print("\n[2/5] Comparaison des modèles...")
    results = train_models(X_train, y_train, X_val, y_val,X_test,y_test)

    # 3. Choisit le meilleur
    print("\n[3/5] Sélection du meilleur modèle...")
    best_name, _ = get_best_model(results,"f1_val")
    print(f" Results : {results}")
    # 4. Tune le meilleur
    print(f"\n[4/5] Tuning {best_name}...")
    best_model, best_params, best_score = tune_model(
        model_name = best_name,
        X_train    = X_train,
        y_train    = y_train,
        X_val      = X_val,
        y_val      = y_val,
        X_test     = X_test,
        y_test     = y_test,
        n_trials   = n_trials
    )
    print(f"Tuning terminé — AUPRC val : {best_score:.4f} ✅")

    # 5. Évalue sur test
    print("\n[5/5] Évaluation finale sur test...")
    evaluate(best_model, X_test, y_test, run_name=best_name + "_final_evaluation")

    # 6. Sauvegarde
    save_artifacts(best_model, scaler, best_params)

    

    return best_model, 


if __name__ == "__main__":
    run_pipeline(n_trials=10)