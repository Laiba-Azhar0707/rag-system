# Chunking
CHUNK_SIZE = 800
CHUNK_OVERLAP = 150
SEPARATORS = ["\n\n", "\n", ". ", " ", ""]

# Embeddings
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# Vector DB
CHROMA_DB_PATH = "./chroma_db"

# Retrieval
TOP_K_RESULTS = 4
SIMILARITY_THRESHOLD = 0.5

# LLM
OLLAMA_API_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "mistral"
OLLAMA_TIMEOUT = 30

# API
API_HOST = "0.0.0.0"
API_PORT = 8000

# Validation
MIN_QUERY_LENGTH = 3
MAX_QUERY_LENGTH = 500
MAX_TOP_K = 10