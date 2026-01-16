"""Memory retrieval for pattern reuse."""
from typing import List, Dict
from src.memory.storage import MemoryStorage
from src.rag.embeddings import EmbeddingGenerator


class MemoryRetriever:
    """Retrieve relevant patterns from memory."""
    
    def __init__(self):
        """Initialize memory retriever."""
        self.storage = MemoryStorage()
        try:
            self.embedder = EmbeddingGenerator()
        except Exception as e:
            print(f"Warning: Failed to initialize EmbeddingGenerator for MemoryRetriever: {e}")
            self.embedder = None
    
    def get_similar_solutions(self, problem_text: str, topic: str, 
                             variables: List[str]) -> List[Dict]:
        """
        Retrieve similar solved problems.
        
        Args:
            problem_text: Current problem text
            topic: Problem topic
            variables: Problem variables
            
        Returns:
            List of similar solutions
        """
        # Get similar problems by topic
        similar = self.storage.get_similar_problems(topic, variables, limit=5)
        
        # Could add semantic similarity here using embeddings
        # For now, return topic-based matches
        
        return similar
    
    def get_ocr_corrections(self, original_text: str) -> List[Dict]:
        """Get OCR corrections for similar text."""
        # This would query the ocr_corrections table
        # For now, return empty list
        return []
