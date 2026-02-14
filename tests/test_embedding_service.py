import sys
import os
import pytest
from unittest.mock import MagicMock, patch

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.embedding_service import get_embedding_model, get_chroma_client, store_embeddings_batched, pipeline_embedding

@patch('src.embedding_service.HuggingFaceEmbeddings')
@patch('src.embedding_service.HF_TOKEN', 'fake_token')
def test_get_embedding_model(mock_hf_embeddings):
    model = get_embedding_model()
    mock_hf_embeddings.assert_called_once()
    assert model is not None

@patch('src.embedding_service.chromadb.HttpClient')
def test_get_chroma_client(mock_http_client):
    """Test ChromaDB client creation."""
    client = get_chroma_client()
    mock_http_client.assert_called_once()
    assert client is not None



@patch('src.embedding_service.get_chroma_client')
@patch('src.embedding_service.pd.read_csv')
@patch('src.embedding_service.store_embeddings_batched')
@patch('src.embedding_service.get_embedding_model')
@patch('src.embedding_service.os.path.exists')
def test_pipeline_embedding(mock_exists, mock_get_model, mock_store, mock_read_csv, mock_get_client):
    
    mock_exists.return_value = True
    
    mock_client = MagicMock()
    mock_collection = MagicMock()
    mock_collection.count.return_value = 0 # Empty collection
    mock_client.get_or_create_collection.return_value = mock_collection
    mock_get_client.return_value = mock_client
    
    # Mock DataFrame
    mock_df = MagicMock()
    mock_df.dropna.return_value = mock_df
    mock_df.__getitem__.return_value.astype.return_value.tolist.return_value = ["clean_text"]
    mock_df.apply.return_value.tolist.return_value = [{"type": "bug"}]
    mock_read_csv.return_value = mock_df
    
    # Run
    pipeline_embedding("fake_path.csv")
    
    # Assertions
    mock_get_client.assert_called()
    mock_read_csv.assert_called_with("fake_path.csv")
    mock_get_model.assert_called_once()
    mock_store.assert_called_once()
