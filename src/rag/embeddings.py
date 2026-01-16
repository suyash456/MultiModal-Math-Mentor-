"""Embedding generation for RAG."""
from typing import List
from sentence_transformers import SentenceTransformer
from src.utils.config import EMBEDDING_MODEL
import os
import torch

# Workaround for PyTorch meta tensor issue
os.environ.setdefault('TRANSFORMERS_OFFLINE', '0')
os.environ.setdefault('HF_HUB_OFFLINE', '0')


class EmbeddingGenerator:
    """Generate embeddings for text chunks."""
    
    def __init__(self):
        """Initialize embedding model."""
        self.model = None
        self._load_model()
    
    def _load_model(self):
        """Load embedding model with error handling."""
        model_name = EMBEDDING_MODEL
        failed_models = []
        
        # Try loading the configured model
        try:
            # Set device explicitly to avoid meta tensor issues
            device = 'cpu'  # Force CPU to avoid GPU/CUDA issues
            # Use trust_remote_code to avoid meta tensor issues
            self.model = SentenceTransformer(
                model_name, 
                device=device,
                trust_remote_code=True
            )
            print(f"Loaded embedding model: {model_name}")
            return
        except Exception as e:
            print(f"Warning: Failed to load {model_name}: {e}")
            failed_models.append(model_name)
        
        # Try fallback models (skip the one that already failed)
        fallback_models = [
            'all-MiniLM-L6-v2',
            'paraphrase-MiniLM-L6-v2',
            'all-mpnet-base-v2'
        ]
        
        # Remove already failed model from fallback list
        fallback_models = [m for m in fallback_models if m not in failed_models]
        
        for fallback_model in fallback_models:
            try:
                device = 'cpu'
                self.model = SentenceTransformer(
                    fallback_model, 
                    device=device,
                    trust_remote_code=True
                )
                print(f"Using fallback model: {fallback_model}")
                return
            except Exception as e:
                print(f"Warning: Failed to load {fallback_model}: {e}")
                failed_models.append(fallback_model)
                continue
        
        # Last resort: try without device specification (only if not already failed)
        if 'all-MiniLM-L6-v2' not in failed_models:
            try:
                # Try without device specification
                self.model = SentenceTransformer('all-MiniLM-L6-v2')
                print("Using fallback model (minimal options): all-MiniLM-L6-v2")
                return
            except Exception as e:
                print(f"Warning: Failed minimal load: {e}")
                failed_models.append('all-MiniLM-L6-v2')
        
        # If all failed, set to None
        print("Critical: All embedding models failed to load")
        print("App will continue without embedding features")
        self.model = None
    
    def embed_text(self, text: str) -> List[float]:
        """Generate embedding for a single text."""
        if self.model is None:
            raise RuntimeError("Embedding model not loaded")
        try:
            return self.model.encode(text, convert_to_numpy=True).tolist()
        except Exception as e:
            print(f"Error generating embedding: {e}")
            raise
    
    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for a batch of texts."""
        if self.model is None:
            raise RuntimeError("Embedding model not loaded")
        try:
            return self.model.encode(texts, convert_to_numpy=True).tolist()
        except Exception as e:
            print(f"Error generating batch embeddings: {e}")
            raise
