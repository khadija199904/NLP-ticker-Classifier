from src.embedding_service import get_vector_count, embedding_pipeline
from src.train_eval import train_and_evaluate 
from src.monitoring import run_monitoring

def main():
    print("=== DEBUT DU PIPELINE BATCH ===")

    # 1. Verification de ChromaDB
    nb_vecteurs = get_vector_count()
    
    if nb_vecteurs > 0:
        print(f" ChromaDB contient déjà {nb_vecteurs} vecteurs. On saute l'étape d'Embedding.")
    else:
        print(" ChromaDB est vide. Lancement de l'Embedding (Etape 2)...")
        embedding_pipeline()

    # 2. Lancement de l'Entraînement (Etape 3)
    print("⚡ Lancement de l'entraînement du modèle...")
    train_and_evaluate()

    # 3. Lancement du Monitoring (Etape 4)
    print("⚡ Génération du rapport de qualité...")
    run_monitoring()

    print("=== PIPELINE TERMINE AVEC SUCCÈS ===")

if __name__ == "__main__":
    main()