import time
import logging
from rag_ollama import ask

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

QUERIES = [
    "What is medical billing?",
    "How do denial appeals work?",
    "What is provider credentialing?",
    "Explain medical coding basics",
    "How to handle insurance claim rejections?"
]

def benchmark():
    logger.info("Running RAG benchmark...\n")
    latencies = []
    for i, query in enumerate(QUERIES, 1):
        start = time.time()
        result = ask(query)
        elapsed = time.time() - start
        latencies.append(elapsed)
        status = "✓" if result.get("answer") and "error" not in result else "✗"
        logger.info(f"[{i}/5] {status} {query[:40]:<40} | {elapsed:.2f}s")
    
    logger.info(f"\nAvg: {sum(latencies)/len(latencies):.2f}s | Min: {min(latencies):.2f}s | Max: {max(latencies):.2f}s")

if __name__ == "__main__":
    benchmark()