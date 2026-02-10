import os
from dotenv import load_dotenv


load_dotenv()

# Configuration des modèles
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
HF_TOKEN = os.getenv("HF_TOKEN")

