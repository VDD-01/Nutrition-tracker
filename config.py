from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

OLLAMA_MODEL = "llama3.2:3b"
OLLAMA_HOST = "http://localhost:11434"

CHROMA_DB_PATH = str(BASE_DIR / "vector_db")
COLLECTION_NAME = "nutrition_data"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

TOP_K_RESULTS = 5
SIMILARITY_THRESHOLD = 0.7

MAX_TOKENS = 512
TEMPERATURE = 0.1
