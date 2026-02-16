import os
import sys
import time
from typing import Any
import pandas as pd
import chromadb
from langchain_huggingface import HuggingFaceEmbeddings 
from langchain_chroma import Chroma

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from config import (
    EMBEDDING_MODEL_NAME, 
    HF_TOKEN, 
    COLLECTION_NAME, 
    CHROMA_HOST, 
    CHROMA_PORT, 
    DATA_CLEANED_PATH
)

def get_embedding_model():
    if not HF_TOKEN:
        raise ValueError("HF_TOKEN est manquant dans les variables d'environnement.")
    print(f"Chargement du modèle d'embeddings : {EMBEDDING_MODEL_NAME}...")
    model_kwargs = {'device': 'cpu'}
    encode_kwargs = {'normalize_embeddings': True} 
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_NAME,
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs
    )



def get_chroma_client():
    return chromadb.HttpClient(host=CHROMA_HOST, port=int(CHROMA_PORT))

def get_vector_count():
    try:
        client = get_chroma_client()
        collection = client.get_collection(name=COLLECTION_NAME)
        return collection.count()
    except Exception as e:
        print(f"Error getting vector count: {e}")
        return 0


def store_embeddings(texts: list[str], embedding_model: Any, metadatas: list[dict]):
    """
    Action : Stockage par lots (batching) dans ChromaDB sans découpage (no chunking).
    """
    if not texts:
        print("Erreur : Aucun texte à stocker.")
        return None

    batch_size = 500  
    total_len = len(texts)
    
    try:
        print(f"Connexion au serveur ChromaDB sur {CHROMA_HOST}:{CHROMA_PORT}...")
        client = get_chroma_client()
        client.heartbeat()

        print(f"Début du stockage de {total_len} documents par lots de {batch_size}...")
        start_time = time.time()
        
        vectorstore = None

        for i in range(0, total_len, batch_size):
            batch_start = i
            batch_end = min(i + batch_size, total_len)
            
            batch_texts = texts[batch_start:batch_end]
            batch_metas = metadatas[batch_start:batch_end]
            
            current_batch_num = (i // batch_size) + 1
            total_batches = (total_len + batch_size - 1) // batch_size
            
            print(f"🚀 Envoi du lot {current_batch_num}/{total_batches} ({len(batch_texts)} docs)...")

            if vectorstore is None:
                # Création de la collection avec le premier lot
                vectorstore = Chroma.from_texts(
                    texts=batch_texts,
                    embedding=embedding_model,
                    metadatas=batch_metas,
                    collection_name=COLLECTION_NAME,
                    client=client
                )
            else:
                # Ajout aux lots suivants
                vectorstore.add_texts(texts=batch_texts, metadatas=batch_metas)

        end_time = time.time()
        execution_time = end_time - start_time
        
        minutes = int(execution_time // 60)
        

        print("-" * 30)
        print(" Indexation terminée avec succès !")
        print(f" Temps total : {minutes}m ({execution_time:.2f} sec)")
    
        print("-" * 30)

        return vectorstore

    except Exception as e:
        print(f" Erreur lors du stockage : {e}")
        return None

def pipeline_embedding(file_path=DATA_CLEANED_PATH):
    """
    Charge le CSV et lance le stockage
    """
    client = get_chroma_client()
    collection = client.get_or_create_collection(name=COLLECTION_NAME)
    
    if collection.count() > 0:
        print(f" ChromaDB contient déjà {collection.count()} documents. On saute l'étape.")
        return
    
    if not os.path.exists(file_path):
        print(f"Fichier {file_path} introuvable.")
        return None

    print(f"Lecture du fichier {file_path}...")
    df = pd.read_csv(file_path).dropna(subset=['text_clean'])

    
    # On prend directement le texte et les colonnes de métadonnées
    texts = df['text_clean'].astype(str).tolist()
    metadatas = df.apply(lambda row: {
        'type': row['type'], 
        'priority': row['priority'], 
        'language': row['language']
    }, axis=1).tolist()

    model = get_embedding_model()
    
    return store_embeddings(texts, model, metadatas)

if __name__ == "__main__":
    pipeline_embedding()