import os
from dotenv import load_dotenv


load_dotenv()

# Configuration des modèles
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
HF_TOKEN = os.getenv("HF_TOKEN")
DATA_CLEANED_PATH = os.getenv("DATA_CLEANED_PATH", "data/data_processed.csv")

# CHROMA CONFIGURATION
CHROMA_HOST = os.getenv("CHROMA_HOST", "localhost")
CHROMA_PORT = os.getenv("CHROMA_PORT", "8080")
COLLECTION_NAME = os.getenv("CHROMA_COLLECTION_NAME", "tickets_collection")
