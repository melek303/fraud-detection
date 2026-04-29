import os
import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import average_precision_score
from dotenv import load_dotenv
from pathlib import Path
from evaluate import compute_metrics, plot_confusion_matrix
from evaluate import evaluate
load_dotenv()

ROOT = Path(__file__).parent.parent.parent
tracking_path = str(ROOT / "mlruns").replace("\\", "/")

mlflow.set_tracking_uri(f"file:///{tracking_path}")
mlflow.set_experiment("fraud-detection")


def get_models(scale_pos_weight: float) -> dict:
    """Retourne les modèles candidats"""
    return {
        "logistic_regression": LogisticRegression(
            max_iter=1000,
            random_state=42
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=100,
            random_state=42,
            class_weight="balanced"  # gère le déséquilibre
        ),
        "xgboost": XGBClassifier(
            random_state=42,
            eval_metric="logloss",
            scale_pos_weight=scale_pos_weight  # gère le déséquilibre
        )
    }


def train_models(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series
) -> dict:
    """
    Entraîne plusieurs modèles et log dans MLflow.
    Retourne les scores AUPRC de chaque modèle.
    """

    # Calcule le ratio pour scale_pos_weight
    scale_pos_weight = y_train.value_counts()[0] / y_train.value_counts()[1]
    print(f"scale_pos_weight : {scale_pos_weight:.2f}")

    models  = get_models(scale_pos_weight)
    results = {}

    for name, model in models.items():
        print(f"\nEntraînement : {name}...")

        with mlflow.start_run(run_name=name):

            # Entraînement
            if name == "xgboost":
                model.fit(X_train, y_train, eval_set=[(X_train,y_train),(X_val, y_val)], verbose=False)
            else:
                model.fit(X_train, y_train)
            train_loss = model.evals_result_["validation_0"]["logloss"] if name == "xgboost" else None
            val_loss   = model.evals_result_["validation_1"]["logloss"] if name == "xgboost" else None
            #plot_training_curve(model, name, train_loss, val_loss)
            if train_loss and val_loss:
                for epoch, (train_score, val_score) in enumerate(zip(train_loss, val_loss),start=1):
                    mlflow.log_metric("train_loss", train_score, step=epoch)
                    mlflow.log_metric("val_loss", val_score, step=epoch)
            # Métriques train
            auprc_train = average_precision_score(
                y_train,
                model.predict_proba(X_train)[:, 1]
            )

            # Log MLflow
            mlflow.log_param("model", name)
            mlflow.log_param("scale_pos_weight", scale_pos_weight)
            mlflow.log_metric("auprc_train", auprc_train)

            # evaluate
            evaluate(model, X_test, y_test)


            metrics = compute_metrics(model, X_test, y_test,tag="val")
            results[name] = {
                "model"      : name,
                "auprc_train": auprc_train,
                
            }
            for metric_name, value in metrics.items():
                results[name][metric_name] = value
            print(f"  auprc_train : {auprc_train:.4f}")
            print(f"  auprc_val   : {metrics["auprc_val"]:.4f}")
            print(f"f1_val       : {results[name]['f1_val']:.4f}")

    return results


def get_best_model(results: dict,metric="auprc_val") -> tuple:
    """Retourne le nom et le modèle avec le meilleur AUPRC val"""
    best_name = max(results, key=lambda x: results[x][metric])
    best_model = results[best_name]["model"]

    print(f"\nMeilleur modèle : {best_name}")
    print(f"AUPRC val       : {results[best_name][metric]:.4f}")

    return best_name, best_model