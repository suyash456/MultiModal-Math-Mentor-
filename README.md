# Multimodal Math Mentor

An end-to-end AI application that reliably solves JEE-style math problems, explains solutions step-by-step, and improves over time through human-in-the-loop feedback and memory.

**Powered by Groq API** - Ultra-fast inference with free tier access!

## Features

- **Multimodal Input**: Accept math problems via image (OCR), audio (ASR), or text
- **Multi-Agent System**: 5 specialized agents working together (Parser, Router, Solver, Verifier, Explainer)
- **RAG Pipeline**: Retrieval-Augmented Generation with curated math knowledge base
- **Human-in-the-Loop (HITL)**: Interactive corrections and validation
- **Memory & Self-Learning**: Learns from past solutions and user feedback

## Installation

1. Clone the repository:
```bash
git clone <https://github.com/suyash456/MultiModal-Math-Mentor-.git>
cd "Multimodal Math Mentor"
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up environment variables:
```bash
cp .env.example .env
# Edit .env and add your GROQ_API_KEY
# Get your API key from https://console.groq.com/
```

4. Initialize the knowledge base:
```bash
python scripts/setup_knowledge_base.py
```

5. Run the application:
```bash
streamlit run app.py
```

## Project Structure

```
.
├── app.py                      # Main Streamlit application
├── src/
│   ├── agents/
│   │   ├── __init__.py
│   │   ├── parser_agent.py     # Parser Agent
│   │   ├── router_agent.py     # Intent Router Agent
│   │   ├── solver_agent.py     # Solver Agent
│   │   ├── verifier_agent.py   # Verifier/Critic Agent
│   │   └── explainer_agent.py  # Explainer/Tutor Agent
│   ├── multimodal/
│   │   ├── __init__.py
│   │   ├── image_processor.py  # Image OCR processing
│   │   ├── audio_processor.py  # Audio ASR processing
│   │   └── text_processor.py   # Text processing
│   ├── rag/
│   │   ├── __init__.py
│   │   ├── knowledge_base.py   # Knowledge base management
│   │   ├── embeddings.py        # Embedding generation
│   │   └── retriever.py        # RAG retrieval
│   ├── memory/
│   │   ├── __init__.py
│   │   ├── storage.py           # Memory storage
│   │   └── retrieval.py         # Memory retrieval
│   └── utils/
│       ├── __init__.py
│       ├── config.py            # Configuration
│       └── hitl.py              # HITL workflow
├── knowledge_base/              # Curated math knowledge documents
├── memory_db/                   # SQLite database for memory
├── scripts/
│   └── setup_knowledge_base.py  # Knowledge base setup script
└── requirements.txt

```

## Usage

1. **Select Input Mode**: Choose Text, Image, or Audio
2. **Provide Input**: Upload/enter your math problem
3. **Review Extraction**: Confirm or edit the extracted text
4. **View Solution**: See step-by-step solution with explanations
5. **Provide Feedback**: Mark solution as correct/incorrect to improve the system

## Architecture

See `ARCHITECTURE.md` for detailed system architecture diagrams and component descriptions.

## Deployment

The app can be deployed to:
- Streamlit Cloud
- HuggingFace Spaces
- Render
- Railway
- Vercel

See `DEPLOYMENT.md` for detailed deployment instructions.

## Scope

Supports:
- Algebra
- Probability
- Basic Calculus (limits, derivatives, optimization)
- Linear Algebra basics

JEE-style difficulty level.

## Evaluation

See `EVALUATION_SUMMARY.md` for comprehensive performance metrics, test results, and system evaluation.

## Documentation

- **[README.md](README.md)** - Project overview and setup
- **[QUICKSTART.md](QUICKSTART.md)** - 5-minute quick start guide
- **[ARCHITECTURE.md](ARCHITECTURE.md)** - System architecture diagrams
- **[DEPLOYMENT.md](DEPLOYMENT.md)** - Deployment instructions
- **[EVALUATION_SUMMARY.md](EVALUATION_SUMMARY.md)** - Performance evaluation
- **[PROJECT_STRUCTURE.md](PROJECT_STRUCTURE.md)** - File organization
- **[IMPLEMENTATION_SUMMARY.md](IMPLEMENTATION_SUMMARY.md)** - Feature summary

## License

MIT
