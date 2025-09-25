# Projet IFOAD - Master Data Science 2024-2025

## Du modèle ML à l'interface utilisateur

Ce projet implémente un cycle complet de développement en apprentissage supervisé :
1. Entraînement d'un modèle de Machine Learning
2. Suivi des performances avec MLflow
3. Exposition du modèle via une API FastAPI
4. Interface utilisateur avec Streamlit

## Structure du projet

```
ML/
├── model/               # Modèle ML et scripts d'entraînement
│   ├── train.py         # Script d'entraînement avec MLflow
│   └── model.joblib     # Modèle entraîné sauvegardé
├── api/                 # API FastAPI
│   └── main.py          # Endpoints API (/health et /predict)
├── ui/                  # Interface utilisateur Streamlit
│   └── app.py           # Application Streamlit
├── mlruns/              # Dossier MLflow (généré automatiquement)
├── requirements.txt     # Dépendances du projet
└── README.md            # Documentation
```

## Installation

1. Cloner le dépôt
2. Installer les dépendances :
   ```
   pip install -r requirements.txt
   ```

## Utilisation

### 1. Entraînement du modèle avec MLflow

```
cd model
python train.py
```

Pour visualiser les résultats dans l'interface MLflow :
```
mlflow ui
```
Puis ouvrir http://localhost:5000 dans un navigateur.

### 2. Lancement de l'API FastAPI

```
cd api
uvicorn main:app --reload
```

L'API sera disponible sur http://localhost:8000

Documentation interactive : http://localhost:8000/docs

### 3. Lancement de l'interface Streamlit

```
cd ui
streamlit run app.py
```

L'interface sera disponible sur http://localhost:8501

## Fonctionnalités

- **Modèle ML** : Classification Random Forest pour prédire la survie au Titanic
- **MLflow** : Suivi des expériences avec différents hyperparamètres
- **FastAPI** : API REST avec endpoints /health et /predict
- **Streamlit** : Interface utilisateur intuitive avec deux modes (API et Local)

## Captures d'écran

Pour le rapport final, n'oubliez pas d'inclure des captures d'écran :
- Interface MLflow montrant les différentes expériences
- Test de l'API avec un exemple de requête/réponse
- Interface Streamlit avec un exemple de prédiction réussie