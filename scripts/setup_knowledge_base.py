"""Setup script for knowledge base."""
import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.rag.knowledge_base import KnowledgeBase

def main():
    """Create default knowledge base."""
    kb = KnowledgeBase()
    
    # Knowledge base is automatically created by RAGRetriever
    # This script can be used to add custom documents
    
    print("Knowledge base setup complete!")
    print(f"Knowledge base directory: {Path('knowledge_base')}")

if __name__ == "__main__":
    main()
