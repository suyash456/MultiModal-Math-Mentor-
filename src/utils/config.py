"""Configuration settings for the Math Mentor application."""
import os
from dotenv import load_dotenv

# Try to load .env file, but don't fail if there's an error
try:
    # Load from current directory first, then check parent
    env_loaded = load_dotenv()
    if not env_loaded:
        # Try loading from project root
        from pathlib import Path
        env_path = Path(__file__).parent.parent.parent / ".env"
        load_dotenv(dotenv_path=env_path)
except Exception as e:
    # If .env loading fails, continue anyway (environment variables can be set directly)
    pass

# API Keys
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "").strip()
# Fallback to OPENAI_API_KEY for backward compatibility
if not GROQ_API_KEY:
    GROQ_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()

# Validate API key
if not GROQ_API_KEY:
    print("WARNING: GROQ_API_KEY not found in environment variables or .env file")

# Model configurations
DEFAULT_LLM_MODEL = "llama-3.1-8b-instant"  # Updated: Current Groq model (fast and efficient)
# Alternative models: "mixtral-8x7b-32768" (larger context), "gemma-7b-it" (alternative)
EMBEDDING_MODEL = "all-MiniLM-L6-v2"  # sentence-transformers model (fast and efficient)

# RAG Configuration
RAG_TOP_K = 5
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50

# Confidence thresholds
OCR_CONFIDENCE_THRESHOLD = 0.7
ASR_CONFIDENCE_THRESHOLD = 0.7
VERIFIER_CONFIDENCE_THRESHOLD = 0.8

# Paths
KNOWLEDGE_BASE_DIR = "knowledge_base"
MEMORY_DB_PATH = "memory_db/math_mentor_memory.db"
VECTOR_STORE_PATH = "vector_store"

# Math topics
SUPPORTED_TOPICS = [
    "algebra",
    "probability",
    "calculus",
    "linear_algebra"
]
