import mlflow
import mlflow.sklearn
import pandas as pd
import optuna
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from evaluate import evaluate,compute_metrics
from preprocess import apply_smote
from pathlib import Path

ROOT = Path(__file__).parent.parent.parent
tracking_path = str(ROOT / "mlruns").replace("\\", "/")

mlflow.set_tracking_uri(f"file:///{tracking_path}")
mlflow.set_experiment("fraud-detection")


def get_search_space(trial, model_name: str, scale_pos_weight: float) -> dict:
    """Retourne l'espace de recherche selon le modèle"""

    if model_name == "xgboost":
        return {
            "use_smote"       :trial.suggest_int("use_smote",0,1),
            "n_estimators"    : trial.suggest_int("n_estimators", 100, 500),
            "max_depth"       : trial.suggest_int("max_depth", 3, 8),
            "learning_rate"   : trial.suggest_float("learning_rate", 0.01, 0.3),
            "subsample"       : trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "scale_pos_weight": trial.suggest_float("scale_pos_weight", scale_pos_weight//2, scale_pos_weight*2),
            "random_state"    : 42,
            "eval_metric"     : "aucpr"
        }

    elif model_name == "random_forest":
        return {
            "use_smote"       :trial.suggest_int("use_smote",0,1),
            "n_estimators": trial.suggest_int("n_estimators", 100, 500),
            "max_depth"   : trial.suggest_int("max_depth", 3, 20),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 10),
            "class_weight": "balanced",
            "random_state": 42
        }

    elif model_name == "logistic_regression":
        return {
            "use_smote"       :trial.suggest_int("use_smote",0,1),
            "C"      : trial.suggest_float("C", 0.001, 10.0),
            "solver" : trial.suggest_categorical("solver", ["lbfgs", "saga"]),
            "max_iter": 1000,
            "random_state": 42
        }


def build_model(model_name: str, params: dict):
    """Construit le modèle avec les paramètres donnés"""

    if model_name == "xgboost":
        return XGBClassifier(**params)
    elif model_name == "random_forest":
        return RandomForestClassifier(**params)
    elif model_name == "logistic_regression":
        return LogisticRegression(**params)


def tune_model(
    model_name: str,        # ← reçoit le nom du meilleur modèle
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    n_trials: int = 50,
    metric= "auprc_val"
):
    """
    Optimise les hyperparamètres du meilleur modèle avec Optuna.
    S'adapte automatiquement au modèle reçu.
    """

    scale_pos_weight = y_train.value_counts()[0] / y_train.value_counts()[1]
    X_train_smote,y_train_smote = apply_smote(X_train, y_train)


    def objective(trial):
        # Espace de recherche adapté au modèle
        params = get_search_space(trial, model_name, scale_pos_weight)
        use_smote = params.pop("use_smote")
        run_name = f"{model_name}_trial_{trial.number}"
        with mlflow.start_run(run_name=run_name):

            model = build_model(model_name, params)
            if use_smote:
                model.fit(X_train_smote, y_train_smote)
            else:
                model.fit(X_train, y_train)

            for param_name, param_value in params.items():
                mlflow.log_param(param_name, param_value)
            mlflow.log_param("use_smote",use_smote)

            evaluate(model, X_test, y_test,)

            metrics = compute_metrics(model, X_val, y_val,tag="val")


        return metrics[metric]

    # Lance l'optimisation
    print(f"Tuning {model_name} — {n_trials} trials...")
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials)

    print(f"\nMeilleurs paramètres :")
    for key, value in study.best_params.items():
        print(f"  {key} : {value}")
    print(f"Meilleur AUPRC val : {study.best_value:.4f}")

    # Réentraîne avec les meilleurs paramètres
    best_params = study.best_params.copy()
    use_smote_best_model = best_params.pop("use_smote")
    best_model = build_model(
        model_name, 
        {**best_params, 
        "random_state": 42,
        "eval_metric": "aucpr"}
    )
    if use_smote_best_model:
        best_model.fit(X_train_smote, y_train_smote)

    else:
        best_model.fit(X_train, y_train)

    return best_model, study.best_params, study.best_value

