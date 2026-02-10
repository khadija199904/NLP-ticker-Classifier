import os
import sys
from langchain_huggingface import HuggingFaceEmbeddings 


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from config import EMBEDDING_MODEL_NAME, HF_TOKEN



def get_embedding_model():
    if not HF_TOKEN:
        raise ValueError("HF_TOKEN est manquant dans les variables d'environnement.")
    print(f"Chargement du modèle d'embeddings : {EMBEDDING_MODEL_NAME}...")
    model_kwargs = {'device': 'cpu'}
    encode_kwargs = {'normalize_embeddings': True}  # normalization 
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_NAME,
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs
    )

if __name__ == "__main__":
    embeddings = get_embedding_model()
    vector = embeddings.embed_query("Ceci est un ticket de support")
    print(vector)