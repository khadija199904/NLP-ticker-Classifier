import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import joblib

from evidently import Report
from evidently.presets import DataDriftPreset, ClassificationPreset

def run_monitoring():
    # Load dataset
    df = pd.read_csv('data/dataset.csv')
    
    # Simple preprocessing 
    df['subject'] = df['subject'].fillna('')
    df['body'] = df['body'].fillna('')
    
    # Split the data into reference and current
    reference_data, current_data = train_test_split(df, test_size=0.3, shuffle=False)

    model = joblib.load('artifacts/models/ticket_classifier.joblib')
    reference_data['prediction'] = model.predict(reference_data[features])
    current_data['prediction'] = model.predict(current_data[features])
    
    

if __name__ == "__main__":
    run_monitoring()
