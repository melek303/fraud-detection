import mlflow
import mlflow.sklearn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    average_precision_score,
    recall_score,
    precision_score,
    f1_score,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score
)
from pathlib import Path

ROOT = Path(__file__).parent.parent.parent
tracking_path = str(ROOT / "mlruns").replace("\\", "/")

mlflow.set_tracking_uri(f"file:///{tracking_path}")
mlflow.set_experiment("fraud-detection")


def compute_metrics(model, X: pd.DataFrame, y: pd.Series,tag) -> dict:
    """Calcule toutes les métriques"""
    y_pred  = model.predict(X)
    y_proba = model.predict_proba(X)[:, 1]

    return {
        f"auprc_{tag}"    : average_precision_score(y, y_proba),
        f"roc_auc_{tag}"  : roc_auc_score(y, y_proba),
        f"recall_{tag}"   : recall_score(y, y_pred),
        f"precision_{tag}": precision_score(y, y_pred),
        f"f1_{tag}"       : f1_score(y, y_pred)
    }


def plot_confusion_matrix(model, X: pd.DataFrame, y: pd.Series) -> plt.Figure:
    """Génère la confusion matrix"""
    y_pred = model.predict(X)
    cm     = confusion_matrix(y, y_pred)

    fig, ax = plt.subplots(figsize=(6, 5))
    sns_labels = [["TN", "FP"], ["FN", "TP"]]

    im = ax.imshow(cm, interpolation='nearest', cmap='Blues')
    plt.colorbar(im)

    for i in range(2):
        for j in range(2):
            ax.text(j, i,
                    f"{sns_labels[i][j]}\n{cm[i,j]}",
                    ha="center", va="center",
                    color="white" if cm[i,j] > cm.max()/2 else "black",
                    fontsize=14)

    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["Normal", "Fraude"])
    ax.set_yticklabels(["Normal", "Fraude"])
    ax.set_xlabel("Prédit")
    ax.set_ylabel("Réel")
    ax.set_title("Confusion Matrix — Test Set")

    plt.tight_layout()
    return fig


def plot_precision_recall_curve(model, X: pd.DataFrame, y: pd.Series) -> plt.Figure:
    """Génère la courbe precision-recall"""
    y_proba = model.predict_proba(X)[:, 1]
    precision, recall, _ = precision_recall_curve(y, y_proba)
    auprc = average_precision_score(y, y_proba)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(recall, precision, color='steelblue', lw=2,
            label=f"AUPRC = {auprc:.4f}")
    ax.axhline(y=y.mean(), color='red', linestyle='--',
               label=f"Baseline = {y.mean():.4f}")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curve — Test Set")
    ax.legend()
    plt.tight_layout()
    return fig

def plot_train_val_loss_curve(train_loss, val_loss) -> plt.Figure:
    """Génère la courbe precision-recall"""
    

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(train_loss, color='steelblue', lw=2,
            label=f"Train_loss")
    
    ax.plot(val_loss, color='red', lw=2,
            label=f"Train_loss")
    
    ax.set_xlabel("Iterations")
    ax.set_ylabel("Log Loss")
    ax.set_title("Training vs Validation Loss")
    ax.legend()
    plt.tight_layout()
    return fig


def evaluate(
    model,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    tag : str ="test",
    log_mlflow = True
):
    """
    Évaluation finale sur le test set.
    Log métriques + artifacts dans MLflow.
    Une seule fois — résultat officiel du projet.
    """

    print(f"\nÉvaluation finale sur X_{tag} et y_{tag}")

    

    # 1. Métriques
    metrics = compute_metrics(model, X_test, y_test,tag=tag)



    # 2. Confusion matrix
    fig_cm = plot_confusion_matrix(model, X_test, y_test)
    

    # 3. Precision-Recall curve
    fig_pr = plot_precision_recall_curve(model, X_test, y_test)
    
    if log_mlflow:
        # 4. Log le modèle final
        mlflow.sklearn.log_model(model)
        for name, value in metrics.items():
            print(f"  {name:12} : {value:.4f}")
            mlflow.log_metric(name, value)
        mlflow.log_figure(fig_cm, "confusion_matrix.png")
        plt.close(fig_cm)
        mlflow.log_figure(fig_pr, "precision_recall_curve.png")
        plt.close(fig_pr)
        print("\nArtifacts sauvegardés dans MLflow ✅")
    
    return metrics,fig_cm,fig_pr
    