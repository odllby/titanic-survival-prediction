import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import mlflow
import mlflow.sklearn
import joblib
import os
import urllib.request

# Configuration de MLflow
mlflow.set_tracking_uri("file:./mlruns")
mlflow.set_experiment("titanic-survival-prediction")

# Téléchargement du dataset Titanic s'il n'existe pas
def download_titanic_dataset():
    if not os.path.exists("titanic.csv"):
        print("Téléchargement du dataset Titanic...")
        url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
        urllib.request.urlretrieve(url, "titanic.csv")
        print("Dataset téléchargé avec succès!")
    return pd.read_csv("titanic.csv")

# Préparation des données
def prepare_data(df):
    # Suppression des colonnes non nécessaires
    df = df.drop(['Name', 'Ticket', 'Cabin', 'PassengerId'], axis=1)
    
    # Gestion des valeurs manquantes
    df['Age'].fillna(df['Age'].median(), inplace=True)
    df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)
    df['Fare'].fillna(df['Fare'].median(), inplace=True)
    
    # Séparation des features et de la target
    X = df.drop('Survived', axis=1)
    y = df['Survived']
    
    # Séparation train/test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    return X_train, X_test, y_train, y_test

# Fonction principale d'entraînement
def train_model(n_estimators=100, max_depth=10, min_samples_split=2):
    # Téléchargement et préparation des données
    df = download_titanic_dataset()
    X_train, X_test, y_train, y_test = prepare_data(df)
    
    # Définition des colonnes numériques et catégorielles
    numeric_features = ['Age', 'Fare', 'SibSp', 'Parch']
    categorical_features = ['Sex', 'Embarked', 'Pclass']
    
    # Préprocesseurs
    numeric_transformer = Pipeline(steps=[
        ('scaler', StandardScaler())
    ])
    
    categorical_transformer = Pipeline(steps=[
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])
    
    # Combinaison des préprocesseurs
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ])
    
    # Création du pipeline complet
    model = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            random_state=42
        ))
    ])
    
    # Début du tracking MLflow
    with mlflow.start_run():
        # Enregistrement des paramètres
        mlflow.log_param("n_estimators", n_estimators)
        mlflow.log_param("max_depth", max_depth)
        mlflow.log_param("min_samples_split", min_samples_split)
        
        # Entraînement du modèle
        model.fit(X_train, y_train)
        
        # Prédictions
        y_pred = model.predict(X_test)
        
        # Calcul des métriques
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        
        # Enregistrement des métriques
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("f1", f1)
        
        # Enregistrement du modèle
        mlflow.sklearn.log_model(model, "model")
        
        print(f"Entraînement terminé avec les paramètres: n_estimators={n_estimators}, max_depth={max_depth}, min_samples_split={min_samples_split}")
        print(f"Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")
        
        # Sauvegarde du meilleur modèle (à adapter selon vos besoins)
        model_dir = "../model"
        os.makedirs(model_dir, exist_ok=True)
        joblib.dump(model, os.path.join(model_dir, "model.joblib"))
        
        return model, accuracy

if __name__ == "__main__":
    # Exécution de plusieurs expériences avec différents hyperparamètres
    experiments = [
        {"n_estimators": 100, "max_depth": 10, "min_samples_split": 2},
        {"n_estimators": 200, "max_depth": 15, "min_samples_split": 2},
        {"n_estimators": 150, "max_depth": 8, "min_samples_split": 5},
        {"n_estimators": 300, "max_depth": 12, "min_samples_split": 3}
    ]
    
    best_accuracy = 0
    best_model = None
    
    for exp in experiments:
        print(f"\nExpérience avec {exp}")
        model, accuracy = train_model(**exp)
        
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_model = model
    
    print(f"\nMeilleure accuracy: {best_accuracy:.4f}")
    
    # Sauvegarde du meilleur modèle
    joblib.dump(best_model, "model.joblib")
    print("Meilleur modèle sauvegardé dans model.joblib")