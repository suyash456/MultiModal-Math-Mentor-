"""Knowledge base management for RAG."""
import os
from typing import List, Dict
from pathlib import Path
from src.utils.config import KNOWLEDGE_BASE_DIR


class KnowledgeBase:
    """Manage the math knowledge base."""
    
    def __init__(self):
        """Initialize knowledge base."""
        self.kb_dir = Path(KNOWLEDGE_BASE_DIR)
        self.kb_dir.mkdir(exist_ok=True)
    
    def get_documents(self) -> List[Dict[str, str]]:
        """
        Load all documents from knowledge base.
        
        Returns:
            List of documents with metadata
        """
        documents = []
        
        if not self.kb_dir.exists():
            return documents
        
        for file_path in self.kb_dir.glob("*.txt"):
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    documents.append({
                        "content": content,
                        "source": str(file_path),
                        "title": file_path.stem
                    })
            except Exception as e:
                print(f"Error loading {file_path}: {e}")
        
        return documents
    
    def add_document(self, title: str, content: str) -> None:
        """Add a document to the knowledge base."""
        file_path = self.kb_dir / f"{title}.txt"
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
