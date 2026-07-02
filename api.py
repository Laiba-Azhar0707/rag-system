from fastapi import FastAPI
from pydantic import BaseModel
from rag import ask

app = FastAPI(
    title="Healthcare RAG API",
    description="Medical billing knowledge base Q&A",
    version="1.0.0"
)

class Query(BaseModel):
    question: str
    top_k: int = 4

@app.post("/ask")
def query_kb(query: Query):
    """Ask a question about medical billing."""
    result = ask(query.question, query.top_k)
    return result

@app.get("/health")
def health():
    return {"status": "ok"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)