import os
import sys
from langchain_huggingface import HuggingFaceEmbeddings 
from langchain_chroma import Chroma
import chromadb
import pandas as pd


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from config import EMBEDDING_MODEL_NAME, HF_TOKEN,COLLECTION_NAME ,CHROMA_HOST,CHROMA_PORT,DATA_CLEANED_PATH



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


def create_embeddings(texts: str):
    if not texts:
        print("Erreur : Aucun texte à Encoder.")
        return None
    embeddings = get_embedding_model().embed_query(texts)
    return embeddings

def store_embeddings(texts :str ,embeddings):
    if not embeddings:
        print("Erreur : Aucun embedding à stocker.")
        return None
    try:
        print(f"Connexion au serveur ChromaDB sur {CHROMA_HOST}:{CHROMA_PORT}...")
        
       
        persistent_client = chromadb.HttpClient(host=CHROMA_HOST, port=CHROMA_PORT)

        vectorstore = Chroma.from_texts(
          texts=texts, 
          embedding=embeddings,
          collection_name=COLLECTION_NAME,
          client=persistent_client
          )
        print(" Base de données vectorielle créée et persistée avec succès.")
       
        return vectorstore
    except Exception as e:
        print(f"Erreur lors de la création de la Vector DB : {e}")
        return None 


def process_csv_and_store(file_path=DATA_CLEANED_PATH, limit=None):
    """
    Chaîne le chargement du CSV et le stockage des embeddings.
    """
    
    if not os.path.exists(file_path):
        print(f"Fichier {file_path} introuvable.")
        return
    
    print(f"Lecture du fichier {file_path}...")
    df = pd.read_csv(file_path)
    
    # Nettoyage rapide des valeurs nulles
    df = df.dropna(subset=['text_clean'])
    if limit:
        df = df.head(limit)
        
    texts = df['text_clean'].tolist()
    metadatas = df[['type', 'priority', 'language']].to_dict(orient='records')
     # vectorization des texs 
    embeddings = create_embeddings(texts)
    store_embeddings(texts, embeddings)
    
    return 
    

if __name__ == "__main__":
    # embeddings = get_embedding_model()
    # vector = embeddings.embed_query("Ceci est un ticket de support")
    # print(vector)
    process_csv_and_store(DATA_CLEANED_PATH)