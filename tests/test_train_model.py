import sys
import os
from unittest.mock import MagicMock, patch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.train_eval import load_data_from_chroma, train_and_evaluate

@patch('src.train_eval.get_chroma_client')
def test_load_data_from_chroma(mock_get_client):
    """Test loading data from ChromaDB."""
    mock_client = MagicMock()
    mock_collection = MagicMock()
    
    mock_collection.count.return_value = 2
    mock_collection.get.return_value = {
        'embeddings': [[0.1, 0.2], [0.3, 0.4]],
        'metadatas': [{'type': 'bug'}, {'type': 'feature'}]
    }
    
    mock_client.get_collection.return_value = mock_collection
    mock_get_client.return_value = mock_client
    
    X, y = load_data_from_chroma()
    
    assert len(X) == 2
    assert len(y) == 2
    assert y == ['bug', 'feature']
    mock_client.get_collection.assert_called_once()

@patch('src.train_eval.load_data_from_chroma')
@patch('src.train_eval.train_test_split')
@patch('src.train_eval.LogisticRegression')
@patch('src.train_eval.joblib.dump')
def test_train_and_evaluate(mock_dump, mock_log_reg, mock_tts, mock_load):
    """Test the training pipeline."""
    
    mock_load.return_value = ([[0.1]], ['bug']) 
    
    mock_tts.return_value = ([[0.1]], [[0.1]], ['bug'], ['bug']) # X_train, X_test, y_train, y_test
    
    mock_clf = MagicMock()
    mock_log_reg.return_value = mock_clf
    mock_clf.predict.return_value = ['bug']
    
    
    train_and_evaluate()
    

    mock_load.assert_called_once()
    mock_tts.assert_called_once()
    mock_clf.fit.assert_called_once()
    mock_clf.predict.assert_called_once()
    mock_dump.assert_called_once()
