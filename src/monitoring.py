import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
import joblib

from evidently import Report, DataDefinition, Dataset, MulticlassClassification
from evidently.presets import DataDriftPreset, ClassificationPreset



import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.embedding_service import get_embedding_model

from config import (
    DATA_CLEANED_PATH, 
    MODEL_SAVE_PATH
)


def run_monitoring():
    print("Préparation du rapport Evidently AI...")
    # Load dataset
    df = pd.read_csv(DATA_CLEANED_PATH).dropna(subset=['text_clean', 'type'])
    
    embedder = get_embedding_model()
    print("fin du get_embedding_model()")
    model = joblib.load(MODEL_SAVE_PATH)
    
    df_sample = df.sample(min(2000, len(df)), random_state=42).copy()
    
    print(" Génération des embeddings pour le monitoring (échantillon)...")
    X_embeddings = embedder.embed_documents(df_sample['text_clean'].tolist())

    print("Calcul des prédictions...")   
    df_sample['prediction'] = model.predict(X_embeddings)
    df_sample['target'] = df_sample['type'] 

      # 6. Séparation en Reference (Passé) et Current (Présent)
    reference_data, current_data = train_test_split(df_sample, test_size=0.5, shuffle=False)




    classification_setup = MulticlassClassification(
        target_column_name='target',
        prediction_column_name='prediction'
    )

    data_definition = DataDefinition(
        classification=[classification_setup],
        text_columns=['text_clean']
    )
    


    reference_dataset = Dataset.from_pandas(reference_data, data_definition=data_definition)
    current_dataset = Dataset.from_pandas(current_data, data_definition=data_definition)


    #  Create a Report
    print("Calcul des métriques de Drift et de Performance...")
    report = Report(metrics=[
        DataDriftPreset(),      
        ClassificationPreset() 
    ])
    

    # Run the report 
    snapshot = report.run(reference_data=reference_dataset, current_data=current_dataset)

    #  Save it as an interactive HTML file
    report_dir = "artifacts/reports"
    os.makedirs(report_dir, exist_ok=True)
    report_path = os.path.join(report_dir, "monitoring_report.html")
    
    snapshot.save_html(report_path)
    print(f" Rapport généré avec succès : {report_path}")

if __name__ == "__main__":
    run_monitoring()
