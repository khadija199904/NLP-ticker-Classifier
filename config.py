import os
from dotenv import load_dotenv
import chromadb



load_dotenv()

# Configuration des modèles
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
HF_TOKEN = os.getenv("HF_TOKEN")
DATA_CLEANED_PATH = os.getenv("DATA_CLEANED_PATH", "data/data_prepared.csv")
MODEL_SAVE_PATH = os.getenv("MODEL_SAVE_PATH", "models/ticket_classifier.joblib")


# CHROMA CONFIGURATION
CHROMA_HOST = os.getenv("CHROMA_HOST", "localhost")
CHROMA_PORT = os.getenv("CHROMA_PORT", "8080")
COLLECTION_NAME = os.getenv("CHROMA_COLLECTION_NAME", "tickets_collection")

if __name__ == "__main__":
  try:
    print(f"Tentative de connexion à ChromaDB sur {CHROMA_HOST}:{CHROMA_PORT}...")
    client = chromadb.HttpClient(host=CHROMA_HOST, port=int(CHROMA_PORT))
    heartbeat = client.heartbeat()
    print(f" Succès ! Serveur opérationnel (Heartbeat: {heartbeat})")
  except Exception as e:
    print(f" Erreur de connexion : {e}")
    print("Assurez-vous que Docker Desktop ou le conteneur ChromaDB est démarré.")