from pathlib import Path
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

try:
    logger.info("Initializing embeddings (sentence-transformers/all-MiniLM-L6-v2)...")
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    logger.info("✓ Embeddings initialized")
except Exception as e:
    logger.error(f"Failed to initialize embeddings: {e}")
    raise

try:
    logger.info("Loading documents from /docs...")
    docs_path = Path("docs")
    
    if not docs_path.exists():
        logger.error(f"Docs folder not found at {docs_path.resolve()}")
        raise FileNotFoundError(f"No 'docs' directory found. Run scraper first.")
    
    documents = []
    file_count = 0
    
    for file in docs_path.glob("*.txt"):
        try:
            with open(file, "r", encoding="utf-8") as f:
                content = f.read()
                if content.strip():
                    documents.append({
                        "content": content,
                        "source": file.name
                    })
                    file_count += 1
                else:
                    logger.warning(f"Empty file skipped: {file.name}")
        except Exception as e:
            logger.warning(f"Failed to read {file.name}: {e}")
            continue
    
    if not documents:
        logger.error("No valid documents found in /docs folder")
        raise ValueError("Document loading failed - no files to process")
    
    logger.info(f"✓ Loaded {file_count} documents")

except FileNotFoundError as e:
    logger.error(f"File error: {e}")
    raise
except Exception as e:
    logger.error(f"Unexpected error loading documents: {e}")
    raise

try:
    logger.info("Chunking documents (chunk_size=800, overlap=150)...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=150,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    
    chunked_docs = []
    for doc in documents:
        try:
            chunks = text_splitter.split_text(doc["content"])
            for chunk in chunks:
                if chunk.strip():  # Only add non-empty chunks
                    chunked_docs.append({
                        "text": chunk,
                        "source": doc["source"]
                    })
        except Exception as e:
            logger.warning(f"Failed to chunk {doc['source']}: {e}")
            continue
    
    if not chunked_docs:
        logger.error("No chunks created - chunking may have failed")
        raise ValueError("Chunking produced no valid results")
    
    logger.info(f"✓ Created {len(chunked_docs)} chunks")

except Exception as e:
    logger.error(f"Chunking failed: {e}")
    raise

try:
    logger.info("Embedding chunks and creating vector database...")
    db = Chroma.from_texts(
        texts=[doc["text"] for doc in chunked_docs],
        metadatas=[{"source": doc["source"]} for doc in chunked_docs],
        embedding=embeddings,
        persist_directory="./chroma_db"
    )
    logger.info("✓ Vector database created: ./chroma_db")

except Exception as e:
    logger.error(f"Failed to create vector database: {e}")
    raise

logger.info("="*60)
logger.info(f"✅ Ingestion Complete!")
logger.info(f"   Documents processed: {len(documents)}")
logger.info(f"   Total chunks: {len(chunked_docs)}")
logger.info(f"   Embedding model: all-MiniLM-L6-v2")
logger.info(f"   Vector database: ./chroma_db")
logger.info("="*60)