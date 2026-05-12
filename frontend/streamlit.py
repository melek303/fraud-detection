import streamlit as st
import requests
import numpy as np
import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.pipeline.preprocess import fetch_data

API_URL = os.getenv("API_URL", "http://localhost:8000")

st.set_page_config(
    page_title="Fraud Detection",
    page_icon="🔍",
    layout="centered"
)

st.title("🔍 Fraud Detection — Live Demo")
st.markdown("Testez si une transaction bancaire est frauduleuse en temps réel.")


# Vérifie que l'API est disponible
try:
    response = requests.get(f"{API_URL}/health")
    st.success("API connectée ✅")
except:
    st.error("API non disponible ❌ — Lance uvicorn d'abord")
    st.stop()

st.divider()

# Mode de test
mode = st.radio(
    "Mode de test",
    ["Transaction manuelle", "Transaction aléatoire du dataset"]
)

if mode == "Transaction aléatoire du dataset":
    query = f"""
    SELECT *
    FROM `fraud-detection-mlops-490823.fraud_detection.transactions`
"""
    df = fetch_data(query)
    
    col1, col2 = st.columns(2)
    with col1:
        show_fraud = st.checkbox("Montrer une fraude")
    with col2:
        if st.button("Générer une transaction"):
            if show_fraud:
                transaction = df[df['Class'] == 1].sample(1).iloc[0]
                st.info("Transaction frauduleuse générée")
            else:
                transaction = df[df['Class'] == 0].sample(1).iloc[0]
                st.info("Transaction normale générée")
            
            st.session_state['transaction'] = transaction.drop('Class').to_dict()

    if 'transaction' in st.session_state:
        st.json(st.session_state['transaction'])

else:
    st.subheader("Entrez les détails de la transaction")
    
    col1, col2 = st.columns(2)
    with col1:
        amount = st.number_input("Montant (€)", min_value=0.0, value=100.0)
    with col2:
        time = st.number_input("Time", min_value=0.0, value=406.0)
    
    # Génère des valeurs aléatoires pour V1-V28
    st.caption("V1-V28 générés aléatoirement (features anonymisées)")
    v_values = {f"V{i}": round(np.random.normal(0, 1), 4) for i in range(1, 29)}
    
    st.session_state['transaction'] = {
        "Time"  : time,
        **v_values,
        "Amount": amount
    }

st.divider()

# Bouton de prédiction
if st.button("🔍 Analyser la transaction", type="primary"):
    if 'transaction' not in st.session_state:
        st.warning("Génère ou entre une transaction d'abord !")
    else:
        with st.spinner("Analyse en cours..."):
            try:
                response = requests.post(
                    f"{API_URL}/predict",
                    json=st.session_state['transaction']
                )
                result = response.json()

                st.divider()
                st.subheader("Résultat")

                col1, col2, col3 = st.columns(3)

                with col1:
                    if result['is_fraud']:
                        st.error("🚨 FRAUDE")
                    else:
                        st.success("✅ LÉGITIME")

                with col2:
                    st.metric(
                        "Probabilité de fraude",
                        f"{result['probability']*100:.2f}%"
                    )

                with col3:
                    risk_colors = {
                        "low"   : "🟢 Faible",
                        "medium": "🟡 Moyen",
                        "high"  : "🔴 Élevé"
                    }
                    st.metric(
                        "Niveau de risque",
                        risk_colors[result['risk_level']]
                    )

                # Barre de progression
                st.progress(result['probability'])

            except Exception as e:
                st.error(f"Erreur : {e}")