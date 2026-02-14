import os
import sys
import chromadb
import pandas as pd
import joblib
import time
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
<<<<<<< HEAD:src/train_eval.py
from sklearn.metrics import classification_report, accuracy_score
from embedding_service import get_chroma_client
=======
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from src.embedding_service import get_chroma_client
>>>>>>> feature/NTC-14-github-actions-cicd:src/train_model.py

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from config import (
    COLLECTION_NAME, 
    MODEL_SAVE_PATH
)

def load_data_from_chroma():
    """
    Récupère les embeddings et les labels directement depuis ChromaDB.
    Évite de recalculer les embeddings à chaque entraînement.
    """
    try:
        print("Chargement des données...")
        client = get_chroma_client()
        collection = client.get_collection(name=COLLECTION_NAME)
    
        total = collection.count()
        X, y = [], []
        batch_size = 5000  

        for i in range(0, total, batch_size):
        
           res = collection.get(include=["embeddings", "metadatas"], limit=batch_size, offset=i)
        
           X.extend(res['embeddings'])
           y.extend([m['type'] for m in res['metadatas']])
           print(f" {len(X)} / {total} récupérés")

        return X, y
        
    except Exception as e:
        print(f"Erreur lors de la récupération des données : {e}")
        return None, None

def train_and_evaluate():
    
    X, y = load_data_from_chroma()
    
    if X is None or len(X) == 0:
        print(" Impossible de continuer : Aucune donnée trouvée dans ChromaDB.")
        return

    #  Séparation des données (80% Train / 20% Test)
   
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    
    print(f" Dataset split : Train={len(X_train)} | Test={len(X_test)}")

    #  Entraînement du modèle
    print("Entraînement du classifieur (Logistic Regression)...")
    clf = LogisticRegression(
        max_iter=1000, 
        random_state=42,
        C=1.0 
    )
    
    start_train = time.time()
    clf.fit(X_train, y_train)
    train_duration = time.time() - start_train
    print(f" Entraînement terminé en {train_duration:.2f}s")

    #  Évaluation
    y_pred = clf.predict(X_test)
    
    print("\n" + "="*40)
    print("RAPPORT DE PERFORMANCE")
    print("="*40)
    print(f"Accuracy Score: {accuracy_score(y_test, y_pred):.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    print("="*40)
    
    
    model_dir = os.path.dirname(MODEL_SAVE_PATH)
    if model_dir: 
        os.makedirs(model_dir, exist_ok=True)
        print(f"📁 Dossier vérifié/créé : {model_dir}")

    joblib.dump(clf, MODEL_SAVE_PATH)
    print(f" Modèle sauvegardé avec succès : {MODEL_SAVE_PATH}")

if __name__ == "__main__":
    train_and_evaluate()