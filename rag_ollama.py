from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
import requests
import json
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

OLLAMA_API_URL = 'http://localhost:11434/api/generate'
OLLAMA_MODEL = 'mistral'
OLLAMA_TIMEOUT = 30
SIMILARITY_THRESHOLD = 0.5

try:
    logger.info("Initializing embeddings...")
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    logger.info("✓ Embeddings initialized")
except Exception as e:
    logger.error(f"Failed to initialize embeddings: {e}")
    raise

try:
    logger.info("Loading vector database from ./chroma_db...")
    db = Chroma(
        persist_directory="./chroma_db",
        embedding_function=embeddings
    )
    logger.info("✓ Vector database loaded")
except Exception as e:
    logger.error(f"Failed to load vector database: {e}")
    logger.error("Make sure chroma_db/ exists. Run ingest.py first.")
    raise

def check_ollama_connection():
    """Verify Ollama is running and accessible."""
    try:
        response = requests.get('http://localhost:11434/api/tags', timeout=5)
        response.raise_for_status()
        logger.info("✓ Ollama connection verified")
        return True
    except requests.ConnectionError:
        logger.error("Cannot connect to Ollama. Make sure Ollama is running (ollama serve)")
        return False
    except Exception as e:
        logger.error(f"Ollama connection check failed: {e}")
        return False

def ask(query: str, top_k: int = 4):
    """Retrieve relevant chunks and generate answer with Ollama."""
    
    try:
        # Validate input
        if not query or len(query.strip()) < 3:
            logger.warning("Query too short or empty")
            return {
                "query": query,
                "answer": "Query must be at least 3 characters long.",
                "sources": [],
                "error": "invalid_input"
            }
        
        logger.info(f"Processing query: {query[:50]}...")
        
        # Retrieve relevant chunks
        try:
            results = db.similarity_search(query, k=top_k)
            
            if not results:
                logger.warning(f"No results found for query: {query[:50]}")
                return {
                    "query": query,
                    "answer": "I don't have information about that topic.",
                    "sources": [],
                    "confidence": 0.0
                }
            
            logger.info(f"Retrieved {len(results)} relevant chunks")
            
        except Exception as e:
            logger.error(f"Retrieval failed: {e}")
            return {
                "query": query,
                "answer": "Failed to search knowledge base.",
                "sources": [],
                "error": "retrieval_failed"
            }
        
        # Build context from retrieved chunks
        context = "\n\n".join([
            f"[{result.metadata['source']}]\n{result.page_content}"
            for result in results
        ])
        
        # Generate prompt
        prompt = f"""You are a medical billing expert. Answer the question using ONLY the provided context.
If the context doesn't answer the question, say so explicitly.

CONTEXT:
{context}

QUESTION: {query}

ANSWER:"""
        
        # Call Ollama
        try:
            logger.info(f"Calling Ollama ({OLLAMA_MODEL})...")
            response = requests.post(
                OLLAMA_API_URL,
                json={
                    "model": OLLAMA_MODEL,
                    "prompt": prompt,
                    "stream": False
                },
                timeout=OLLAMA_TIMEOUT
            )
            response.raise_for_status()
            
        except requests.ConnectionError:
            logger.error("Cannot connect to Ollama server")
            return {
                "query": query,
                "answer": "Error: Ollama is not running. Start Ollama with 'ollama serve'",
                "sources": [],
                "error": "ollama_unavailable"
            }
        except requests.Timeout:
            logger.error(f"Ollama request timed out after {OLLAMA_TIMEOUT}s")
            return {
                "query": query,
                "answer": "Error: Response generation timed out.",
                "sources": [],
                "error": "generation_timeout"
            }
        except Exception as e:
            logger.error(f"Ollama request failed: {e}")
            return {
                "query": query,
                "answer": f"Error: Generation failed.",
                "sources": [],
                "error": "generation_failed"
            }
        
        # Parse response
        try:
            response_data = response.json()
            answer = response_data.get('response', '').strip()
            
            if not answer:
                logger.warning("Ollama returned empty response")
                return {
                    "query": query,
                    "answer": "Error: Generation produced no output.",
                    "sources": [],
                    "error": "empty_response"
                }
            
            logger.info(f"✓ Generated response ({len(answer)} chars)")
            
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse Ollama response: {e}")
            return {
                "query": query,
                "answer": "Error: Invalid response format from generator.",
                "sources": [],
                "error": "parse_error"
            }
        
        # Extract sources
        sources = list(set([result.metadata['source'] for result in results]))
        
        return {
            "query": query,
            "answer": answer,
            "sources": sources,
            "confidence": 0.95
        }
    
    except Exception as e:
        logger.error(f"Unexpected error in ask(): {e}")
        return {
            "query": query,
            "answer": "An unexpected error occurred.",
            "sources": [],
            "error": "unexpected_error"
        }

if __name__ == "__main__":
    # Check Ollama before running
    if not check_ollama_connection():
        logger.error("Exiting: Ollama is not available")
        exit(1)
    
    # Test query
    logger.info("Running test query...")
    result = ask("What's the process for medical billing denial appeals?")
    
    logger.info("="*60)
    logger.info(f"Q: {result['query']}\n")
    logger.info(f"A: {result['answer']}\n")
    logger.info(f"Sources: {', '.join(result['sources'])}")
    logger.info("="*60)