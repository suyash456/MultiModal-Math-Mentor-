"""RAG retriever for relevant context."""
import os
import pickle
from typing import List, Dict, Tuple
import faiss
import numpy as np
from pathlib import Path
from src.rag.knowledge_base import KnowledgeBase
from src.rag.embeddings import EmbeddingGenerator
from src.utils.config import RAG_TOP_K, CHUNK_SIZE, CHUNK_OVERLAP, VECTOR_STORE_PATH


class RAGRetriever:
    """Retrieve relevant context using vector similarity."""
    
    def __init__(self):
        """Initialize RAG retriever."""
        self.kb = KnowledgeBase()
        try:
            self.embedder = EmbeddingGenerator()
        except Exception as e:
            print(f"Warning: Failed to initialize EmbeddingGenerator: {e}")
            self.embedder = None
        self.index = None
        self.chunks = []
        self.metadata = []
        self.vector_store_path = Path(VECTOR_STORE_PATH)
        self.vector_store_path.mkdir(exist_ok=True)
        
        if self.embedder is not None:
            self._build_index()
        else:
            print("Warning: RAG retriever initialized without embeddings. Some features may not work.")
    
    def _chunk_text(self, text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> List[str]:
        """Split text into overlapping chunks."""
        chunks = []
        words = text.split()
        
        for i in range(0, len(words), chunk_size - overlap):
            chunk = " ".join(words[i:i + chunk_size])
            chunks.append(chunk)
        
        return chunks
    
    def _build_index(self):
        """Build or load FAISS index."""
        index_path = self.vector_store_path / "faiss_index.pkl"
        chunks_path = self.vector_store_path / "chunks.pkl"
        metadata_path = self.vector_store_path / "metadata.pkl"
        
        # Try to load existing index
        if index_path.exists() and chunks_path.exists() and metadata_path.exists():
            try:
                with open(index_path, 'rb') as f:
                    self.index = pickle.load(f)
                with open(chunks_path, 'rb') as f:
                    self.chunks = pickle.load(f)
                with open(metadata_path, 'rb') as f:
                    self.metadata = pickle.load(f)
                print(f"Loaded existing index with {len(self.chunks)} chunks")
                return
            except Exception as e:
                print(f"Error loading index: {e}")
        
        # Build new index
        print("Building new index...")
        documents = self.kb.get_documents()
        
        if not documents:
            print("No documents in knowledge base. Creating default index.")
            self._create_default_knowledge_base()
            documents = self.kb.get_documents()
        
        all_chunks = []
        all_metadata = []
        
        for doc in documents:
            chunks = self._chunk_text(doc["content"])
            all_chunks.extend(chunks)
            all_metadata.extend([{
                "source": doc["source"],
                "title": doc["title"]
            }] * len(chunks))
        
        if not all_chunks:
            # Create empty index
            dimension = 384  # Default for all-MiniLM-L6-v2
            self.index = faiss.IndexFlatL2(dimension)
            self.chunks = []
            self.metadata = []
            return
        
        # Check if embedder is available
        if self.embedder is None:
            print("Warning: Embedder not available, cannot build index")
            # Create empty index
            dimension = 384
            self.index = faiss.IndexFlatL2(dimension)
            self.chunks = []
            self.metadata = []
            return
        
        # Generate embeddings
        try:
            embeddings = self.embedder.embed_batch(all_chunks)
            embeddings_array = np.array(embeddings).astype('float32')
        except Exception as e:
            print(f"Error generating embeddings: {e}")
            # Create empty index
            dimension = 384
            self.index = faiss.IndexFlatL2(dimension)
            self.chunks = []
            self.metadata = []
            return
        
        # Create FAISS index
        dimension = embeddings_array.shape[1]
        self.index = faiss.IndexFlatL2(dimension)
        self.index.add(embeddings_array)
        
        self.chunks = all_chunks
        self.metadata = all_metadata
        
        # Save index
        try:
            with open(index_path, 'wb') as f:
                pickle.dump(self.index, f)
            with open(chunks_path, 'wb') as f:
                pickle.dump(self.chunks, f)
            with open(metadata_path, 'wb') as f:
                pickle.dump(self.metadata, f)
            print(f"Saved index with {len(self.chunks)} chunks")
        except Exception as e:
            print(f"Error saving index: {e}")
    
    def _create_default_knowledge_base(self):
        """Create default knowledge base documents."""
        default_docs = {
            "algebra_basics": """Algebra Basics

Key Concepts:
- Variables: x, y, z represent unknown values
- Equations: expressions with = sign
- Solving: isolate the variable

Common Operations:
- Addition/Subtraction: x + 5 = 10 → x = 5
- Multiplication/Division: 3x = 15 → x = 5
- Quadratic Formula: x = (-b ± √(b²-4ac)) / 2a

Example: Solve 2x + 3 = 11
Step 1: Subtract 3 from both sides → 2x = 8
Step 2: Divide by 2 → x = 4""",
            
            "probability_basics": """Probability Basics

Key Concepts:
- Probability: P(event) = favorable outcomes / total outcomes
- Range: 0 ≤ P ≤ 1
- Complement: P(not A) = 1 - P(A)

Common Formulas:
- P(A and B) = P(A) × P(B) if independent
- P(A or B) = P(A) + P(B) - P(A and B)
- Conditional: P(A|B) = P(A and B) / P(B)

Example: Probability of rolling 6 on a die
P(6) = 1/6 ≈ 0.167""",
            
            "calculus_basics": """Calculus Basics

Derivatives:
- Definition: f'(x) = lim(h→0) [f(x+h) - f(x)] / h
- Power rule: d/dx(xⁿ) = nxⁿ⁻¹
- Product rule: (fg)' = f'g + fg'
- Quotient rule: (f/g)' = (f'g - fg') / g²
- Chain rule: (f(g(x)))' = f'(g(x)) × g'(x)

Integrals:
- Antiderivative: ∫f(x)dx = F(x) + C
- Power rule: ∫xⁿdx = xⁿ⁺¹/(n+1) + C

Limits:
- Evaluate by substitution, factoring, or L'Hôpital's rule""",
            
            "linear_algebra_basics": """Linear Algebra Basics

Matrices:
- Addition: element-wise
- Multiplication: row × column
- Determinant: det(A) for 2×2: ad - bc
- Inverse: A⁻¹ = (1/det(A)) × adj(A)

Vectors:
- Dot product: a·b = Σaᵢbᵢ
- Cross product: a×b (3D only)

Systems of Equations:
- Use elimination or substitution
- Matrix form: Ax = b → x = A⁻¹b"""
        }
        
        for title, content in default_docs.items():
            self.kb.add_document(title, content)
    
    def retrieve(self, query: str, top_k: int = RAG_TOP_K) -> List[Dict[str, str]]:
        """
        Retrieve relevant chunks for a query.
        
        Args:
            query: Search query
            top_k: Number of results to return
            
        Returns:
            List of relevant chunks with metadata
        """
        if self.embedder is None:
            print("Warning: Embedder not available, returning empty results")
            return []
        
        if self.index is None or len(self.chunks) == 0:
            return []
        
        try:
            # Generate query embedding
            query_embedding = self.embedder.embed_text(query)
            query_vector = np.array([query_embedding]).astype('float32')
            
            # Search
            k = min(top_k, len(self.chunks))
            distances, indices = self.index.search(query_vector, k)
            
            # Format results
            results = []
            for i, idx in enumerate(indices[0]):
                if idx < len(self.chunks):
                    results.append({
                        "content": self.chunks[idx],
                        "source": self.metadata[idx]["source"],
                        "title": self.metadata[idx]["title"],
                        "distance": float(distances[0][i])
                    })
            
            return results
        except Exception as e:
            print(f"Retrieval error: {e}")
            return []
