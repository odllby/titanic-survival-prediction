from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import numpy as np
import os
import sys

# Ajout du chemin parent pour l'importation
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Chargement du modèle
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "model", "model.joblib")
try:
    model = joblib.load(MODEL_PATH)
    print(f"Modèle chargé avec succès depuis: {MODEL_PATH}")
except Exception as e:
    print(f"Erreur lors du chargement du modèle: {str(e)}")
    model = None

app = FastAPI(title="Titanic Survival Prediction API")

# Définition du schéma de données d'entrée
class TitanicFeatures(BaseModel):
    Pclass: int
    Sex: str
    Age: float
    SibSp: int
    Parch: int
    Fare: float
    Embarked: str

# Définition du schéma de données de sortie
class PredictionResponse(BaseModel):
    survival_prediction: int
    survival_probability: float
    message: str

@app.get("/health")
def health_check():
    """Vérification de l'état de l'API"""
    return {"status": "healthy", "model_loaded": model is not None}

@app.post("/predict", response_model=PredictionResponse)
def predict(features: TitanicFeatures):
    """Prédiction de la survie d'un passager du Titanic"""
    try:
        # Vérification que le modèle est chargé
        if model is None:
            raise HTTPException(status_code=500, detail="Le modèle n'est pas chargé")
            
        # Conversion des données en DataFrame pour le préprocesseur
        import pandas as pd
        input_df = pd.DataFrame([{
            'Pclass': features.Pclass,
            'Sex': features.Sex,
            'Age': features.Age,
            'SibSp': features.SibSp,
            'Parch': features.Parch,
            'Fare': features.Fare,
            'Embarked': features.Embarked
        }])
        
        print(f"Données reçues: {input_df.to_dict('records')}")
        
        # Extraction du pipeline et du classificateur
        # Si le modèle est un pipeline scikit-learn
        if hasattr(model, 'steps'):
            # Prédiction directe avec le pipeline
            prediction = model.predict(input_df)[0]
            probability = model.predict_proba(input_df)[0][1]
        else:
            # Si c'est juste un classificateur
            prediction = model.predict(input_df.values)[0]
            probability = model.predict_proba(input_df.values)[0][1]
        
        # Préparation de la réponse
        result = {
            "survival_prediction": int(prediction),
            "survival_probability": float(probability),
            "message": "Le passager aurait survécu." if prediction == 1 else "Le passager n'aurait pas survécu."
        }
        
        print(f"Prédiction réussie: {result}")
        return result
    
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        print(f"Erreur de prédiction: {str(e)}\n{error_details}")
        raise HTTPException(status_code=500, detail=f"Erreur de prédiction: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)