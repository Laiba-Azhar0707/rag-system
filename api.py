from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, validator
from rag_ollama import ask, check_ollama_connection
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Medical Billing RAG API",
    description="Retrieval-augmented generation system for medical billing knowledge base",
    version="1.0.0"
)

# Input validation model
class Query(BaseModel):
    question: str = Field(
        ...,
        min_length=3,
        max_length=500,
        description="The question to ask the knowledge base"
    )
    top_k: int = Field(
        default=4,
        ge=1,
        le=10,
        description="Number of relevant chunks to retrieve (1-10)"
    )
    
    @validator('question')
    def validate_question(cls, v):
        """Ensure question is not just whitespace."""
        if not v.strip():
            raise ValueError("Question cannot be empty or whitespace")
        return v.strip()

# Response model
class QueryResponse(BaseModel):
    query: str
    answer: str
    sources: list
    confidence: float = 0.0
    error: str = None

# Health check endpoint
@app.get("/health")
def health_check():
    """Check if API and dependencies are available."""
    try:
        ollama_ok = check_ollama_connection()
        
        return {
            "status": "healthy" if ollama_ok else "degraded",
            "ollama": "connected" if ollama_ok else "disconnected",
            "api": "operational"
        }
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return {
            "status": "unhealthy",
            "error": str(e)
        }

# Main query endpoint
@app.post("/ask", response_model=QueryResponse)
def query_knowledge_base(query: Query):
    """Query the medical billing knowledge base and get an answer."""
    
    try:
        logger.info(f"Received query: {query.question[:50]}...")
        
        # Call RAG pipeline
        result = ask(query.question, top_k=query.top_k)
        
        # Check for errors from RAG
        if "error" in result and result["error"]:
            logger.warning(f"RAG error: {result['error']}")
            raise HTTPException(
                status_code=500,
                detail=result.get("answer", "Generation failed")
            )
        
        logger.info(f"✓ Query processed successfully")
        
        return QueryResponse(
            query=result["query"],
            answer=result["answer"],
            sources=result["sources"],
            confidence=result.get("confidence", 0.0),
            error=result.get("error")
        )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error in query endpoint: {e}")
        raise HTTPException(
            status_code=500,
            detail="An unexpected error occurred while processing your query"
        )

# Root endpoint
@app.get("/")
def root():
    """API information."""
    return {
        "name": "Medical Billing RAG API",
        "version": "1.0.0",
        "description": "Retrieval-augmented generation system for medical billing Q&A",
        "endpoints": {
            "health": "GET /health",
            "ask": "POST /ask",
            "docs": "GET /docs"
        }
    }

# Startup event
@app.on_event("startup")
async def startup_event():
    """Verify Ollama connection on startup."""
    logger.info("Starting Medical Billing RAG API...")
    
    if not check_ollama_connection():
        logger.warning("Ollama is not running. The API will not function properly.")
        logger.warning("Start Ollama with: ollama serve")
    else:
        logger.info("✓ All dependencies initialized successfully")

# Shutdown event
@app.on_event("shutdown")
async def shutdown_event():
    """Log shutdown."""
    logger.info("Shutting down Medical Billing RAG API")

if __name__ == "__main__":
    import uvicorn
    
    logger.info("Launching API server on http://0.0.0.0:8000")
    logger.info("API documentation available at http://localhost:8000/docs")
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info"
    )