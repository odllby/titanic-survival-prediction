import streamlit as st
import requests
import pandas as pd
import joblib
import sys
import os

# Configuration de la page
st.set_page_config(
    page_title="Prédiction de survie au Titanic",
    page_icon="🚢",
    layout="centered"
)

# Titre et description
st.title("🚢 Prédiction de survie au Titanic")
st.markdown("""
Cette application vous permet de prédire si un passager aurait survécu au naufrage du Titanic
en fonction de ses caractéristiques personnelles.
""")

# URL de l'API
API_URL = "http://localhost:8000"

# Fonction pour appeler l'API
def predict_api(features):
    response = requests.post(f"{API_URL}/predict", json=features)
    return response.json()

# Fonction pour prédire directement avec le modèle local
def predict_local(features):
    try:
        # Chargement du modèle
        model_path = "../model/model.joblib"
        model = joblib.load(model_path)
        
        # Préparation des données en DataFrame pandas
        input_df = pd.DataFrame([features])
        
        # Prédiction
        if hasattr(model, 'steps'):
            # Si c'est un pipeline
            prediction = model.predict(input_df)[0]
            probability = model.predict_proba(input_df)[0][1]
        else:
            # Si c'est juste un classificateur
            prediction = model.predict(input_df.values)[0]
            probability = model.predict_proba(input_df.values)[0][1]
        
        return {
            "survival_prediction": int(prediction),
            "survival_probability": float(probability),
            "message": "Le passager aurait survécu." if prediction == 1 else "Le passager n'aurait pas survécu."
        }
    except Exception as e:
        import traceback
        print(f"Erreur dans predict_local: {str(e)}")
        print(traceback.format_exc())
        raise e

# Mode de prédiction (API ou Local)
prediction_mode = st.sidebar.radio(
    "Mode de prédiction",
    ["API", "Local (sans API)"]
)

# Formulaire pour les caractéristiques du passager
with st.form("prediction_form"):
    st.subheader("Caractéristiques du passager")
    
    col1, col2 = st.columns(2)
    
    with col1:
        pclass = st.selectbox(
            "Classe",
            options=[1, 2, 3],
            format_func=lambda x: f"{x} - {'Première' if x == 1 else 'Deuxième' if x == 2 else 'Troisième'}"
        )
        
        sex = st.radio(
            "Sexe",
            options=["male", "female"],
            format_func=lambda x: "Homme" if x == "male" else "Femme"
        )
        
        age = st.slider(
            "Âge",
            min_value=0.5,
            max_value=80.0,
            value=30.0,
            step=0.5
        )
    
    with col2:
        sibsp = st.slider(
            "Nombre de frères/sœurs/conjoints à bord",
            min_value=0,
            max_value=8,
            value=0
        )
        
        parch = st.slider(
            "Nombre de parents/enfants à bord",
            min_value=0,
            max_value=6,
            value=0
        )
        
        fare = st.slider(
            "Prix du billet (£)",
            min_value=0.0,
            max_value=512.0,
            value=32.0,
            step=1.0
        )
        
        embarked = st.selectbox(
            "Port d'embarquement",
            options=["C", "Q", "S"],
            format_func=lambda x: "Cherbourg" if x == "C" else "Queenstown" if x == "Q" else "Southampton"
        )
    
    submit_button = st.form_submit_button("Prédire")

# Prédiction
if submit_button:
    # Préparation des données
    features = {
        "Pclass": pclass,
        "Sex": sex,
        "Age": age,
        "SibSp": sibsp,
        "Parch": parch,
        "Fare": fare,
        "Embarked": embarked
    }
    
    with st.spinner("Prédiction en cours..."):
        try:
            if prediction_mode == "API":
                # Appel à l'API
                result = predict_api(features)
                st.success("Prédiction via API réussie!")
            else:
                # Prédiction locale
                result = predict_local(features)
                st.success("Prédiction locale réussie!")
            
            # Affichage du résultat
            st.subheader("Résultat de la prédiction")
            
            # Probabilité de survie
            survival_prob = result["survival_probability"] * 100
            
            # Affichage visuel
            if result["survival_prediction"] == 1:
                st.markdown(f"### ✅ **{result['message']}**")
                st.progress(survival_prob / 100)
                st.markdown(f"**Probabilité de survie: {survival_prob:.1f}%**")
            else:
                st.markdown(f"### ❌ **{result['message']}**")
                st.progress(1 - (survival_prob / 100))
                st.markdown(f"**Probabilité de décès: {100 - survival_prob:.1f}%**")
            
            # Détails techniques
            with st.expander("Détails techniques"):
                st.json(result)
                st.markdown("**Caractéristiques utilisées:**")
                st.json(features)
        
        except Exception as e:
            st.error(f"Erreur lors de la prédiction: {str(e)}")
            if prediction_mode == "API":
                st.warning("Assurez-vous que l'API FastAPI est en cours d'exécution sur http://localhost:8000")

# Informations sur le projet
with st.sidebar:
    st.subheader("À propos du projet")
    st.markdown("""
    Ce projet fait partie du Master en Data Science IFOAD 2024-2025.
    
    **Pipeline complet:**
    1. Entraînement du modèle avec MLflow
    2. API FastAPI
    3. Interface utilisateur Streamlit
    
    **Technologies utilisées:**
    - Scikit-learn
    - MLflow
    - FastAPI
    - Streamlit
    """)