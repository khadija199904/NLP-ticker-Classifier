import os
import sys
from typing import Any
from langchain_huggingface import HuggingFaceEmbeddings 
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
import chromadb
import pandas as pd


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from config import EMBEDDING_MODEL_NAME, HF_TOKEN,COLLECTION_NAME ,CHROMA_HOST,CHROMA_PORT,DATA_CLEANED_PATH


def perform_chunking(df):
    """Découpe les textes en morceaux de 400 caractères (~100 mots)."""

     # Configuration du splitter
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=400,
        chunk_overlap=50,
        separators=[" "]
    )


    print(" Chunking en mémoire...")
    texts, metadatas = [], []
    for _, row in df.iterrows():
        chunks = splitter.split_text(str(row['text_clean']))
        for chunk in chunks:
            texts.append(chunk)
            metadatas.append({
                'type': row['type'], 
                'priority': row['priority'], 
                'language': row['language']
            })
    return texts, metadatas

def get_embedding_model():
    if not HF_TOKEN:
        raise ValueError("HF_TOKEN est manquant dans les variables d'environnement.")
    print(f"Chargement du modèle d'embeddings : {EMBEDDING_MODEL_NAME}...")
    model_kwargs = {'device': 'cpu'}
    encode_kwargs = {'normalize_embeddings': True}  # normalization 
    model_embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_NAME,
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs
    )
    return model_embeddings



def store_embeddings(texts: list[str], embedding_model: Any, metadatas: list[dict] | None = None):
    """
    Action : Stockage dans ChromaDB via HttpClient.
    """
    if not texts:
        print("Erreur : Aucun texte à stocker.")
        return None
    try:
        print(f"Connexion au serveur ChromaDB sur {CHROMA_HOST}:{CHROMA_PORT}...")
        
        client = chromadb.HttpClient(host=CHROMA_HOST, port=int(CHROMA_PORT))
        client.heartbeat()

        batch_size = 500
        vectorstore = None
        
        total_len = len(texts)
        total_batches = (len(texts) + batch_size - 1) // batch_size
        for i in range(0, total_len, batch_size):
            end_idx = min(i + batch_size, total_len)
            batch_txt = texts[i:end_idx]
            batch_meta = metadatas[i:end_idx] if metadatas is not None else None


            current_batch = i // batch_size + 1
            print(f"🚀 Lot {current_batch}/{total_batches}...")

            if vectorstore is None:
                vectorstore = Chroma.from_texts(
                    texts=batch_txt, 
                    embedding=embedding_model, 
                    metadatas=batch_meta,
                    collection_name=COLLECTION_NAME, 
                    client=client
                )
            else:
                vectorstore.add_texts(texts=batch_txt, metadatas=batch_meta)
    
        print("Transfert terminé avec succès !")

       
        return vectorstore
    except Exception as e:
        print(f"Erreur lors du stockage dans ChromaDB : {e}")
        return None 

def process_and_store(file_path=DATA_CLEANED_PATH, limit=None):
    """
    Chaîne le chargement du CSV et le stockage.
    """
    if os.path.exists(file_path):
        print(f"Lecture du fichier {file_path}...")
        df = pd.read_csv(file_path).dropna(subset=['text_clean'])
    
        #  Découper en morceaux (Chunks)
        texts, metadatas = perform_chunking(df)
        model = get_embedding_model()
    
        return store_embeddings(texts, model, metadatas=metadatas)
    else: 
        print(f"Fichier {file_path} introuvable.")
        return None

if __name__ == "__main__":
    
    process_and_store(DATA_CLEANED_PATH)