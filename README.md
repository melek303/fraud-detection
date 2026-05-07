# 🔍 Fraud Detection — MLOps Pipeline

Système de détection de fraude sur des transactions bancaires, avec pipeline d'entraînement complet, tuning automatique des hyperparamètres et tracking des expériences via MLflow.

---

## 📁 Structure du projet

```
├── src/
│   └── pipeline/
│       ├── __init__.py
│       ├── preprocess.py        # Fetch, split, scale, SMOTE
│       ├── train.py             # Comparaison de modèles
│       ├── tune.py              # Optimisation Optuna
│       ├── evaluate.py          # Métriques + visualisations + MLflow
│       └── training_pipeline.py # Orchestrateur principal
├── notebooks/
│   └── 03_best_model.ipynb     # Évaluation interactive du meilleur modèle
├── models/
│   ├── best_model.pkl
│   ├── scaler.pkl
│   └── best_parameters.pkl
└── mlruns/                      # Tracking MLflow local
```

---

## ⚙️ Pipeline d'entraînement

Le pipeline complet est orchestré par `training_pipeline.py` et se déroule en 5 étapes :

### 1. Preprocessing (`preprocess.py`)
- Récupération des données depuis **BigQuery** (`fraud-detection-mlops-490823.fraud_detection.transactions`)
- Split stratifié **train / val / test** (80% / 10% / 10%)
- Scaling de `Amount` et `Time` avec `RobustScaler` (fit uniquement sur le train)
- Application optionnelle de **SMOTE** (uniquement sur le train)

### 2. Comparaison de modèles (`train.py`)
Trois modèles sont entraînés et comparés :

| Modèle | Gestion du déséquilibre |
|---|---|
| Logistic Regression | — |
| Random Forest | `class_weight="balanced"` |
| XGBoost | `scale_pos_weight` |

Le meilleur modèle est sélectionné selon l'**AUPRC sur le val set** (configurable).

### 3. Tuning des hyperparamètres (`tune.py`)
- Optimisation via **Optuna** (`n_trials` configurable, défaut : 50)
- L'espace de recherche s'adapte automatiquement au modèle sélectionné
- `use_smote` est lui-même un hyperparamètre optimisé (0 ou 1)
- Chaque trial est loggé dans MLflow

### 4. Évaluation (`evaluate.py`)
Métriques calculées sur train, val et test :

- **AUPRC** (Area Under Precision-Recall Curve) — métrique principale
- **ROC-AUC**
- Recall, Precision, F1-score

Artefacts générés et loggés dans MLflow :
- Confusion matrix
- Courbe Precision-Recall
- Courbe Train vs Validation Loss (XGBoost uniquement)

### 5. Sauvegarde
Le modèle final, le scaler et les meilleurs paramètres sont sauvegardés dans `models/`.

---

## 🚀 Lancer le pipeline

```bash
cd src/pipeline
python training_pipeline.py
```

Options configurables directement dans `run_pipeline()` :

```python
run_pipeline(
    n_trials=20,       # Nombre de trials Optuna
    metric="auprc_val" # Métrique de sélection du meilleur modèle
)
```

---

## 📊 Tracking MLflow

Les expériences sont trackées localement dans `mlruns/` sous l'expérience `fraud-detection`.

```bash
mlflow ui
```

Puis ouvrir [http://localhost:5000](http://localhost:5000).

Chaque run contient :
- Les hyperparamètres
- Les métriques train / val / test
- Les artefacts visuels (confusion matrix, courbes)

---

## 📓 Notebook d'évaluation

Le notebook `03_best_model.ipynb` permet de recharger le modèle sauvegardé et de relancer une évaluation sur le test set sans relancer le pipeline complet :

```python
best_model = joblib.load("../models/best_model.pkl")
scaler     = joblib.load("../models/scaler.pkl")

metrics, confusion_matrix, pr_curve = evaluate(
    best_model, X_test, y_test,
    log_mlflow=False,
    tag="test"
)
```

---

## 🛠️ Stack technique

| Composant | Outil |
|---|---|
| Data source | Google BigQuery |
| Preprocessing | scikit-learn, imbalanced-learn (SMOTE) |
| Modèles | Logistic Regression, Random Forest, XGBoost |
| Tuning | Optuna |
| Tracking | MLflow |
| Sérialisation | joblib |
| Environnement | python-dotenv |

---

## 📦 Installation

```bash
pip install -r requirements.txt
```

Dépendances principales :

```
scikit-learn
xgboost
imbalanced-learn
optuna
mlflow
google-cloud-bigquery
joblib
python-dotenv
pandas
matplotlib
```

> **Prérequis** : Avoir configuré l'authentification Google Cloud (`gcloud auth application-default login`) pour accéder à BigQuery.