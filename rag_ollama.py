from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
import requests
import json

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
db = Chroma(
    persist_directory="./chroma_db",
    embedding_function=embeddings
)

def ask(query: str, top_k: int = 4):
    """Retrieve relevant chunks and generate answer with Ollama."""
    
    results = db.similarity_search(query, k=top_k)
    
    context = "\n\n".join([
        f"[{result.metadata['source']}]\n{result.page_content}"
        for result in results
    ])
    
    prompt = f"""You are a medical billing expert. Answer using only the provided context.
If context doesn't answer the question, say so explicitly.

CONTEXT:
{context}

QUESTION: {query}

ANSWER:"""
    
    response = requests.post('http://localhost:11434/api/generate', json={
        "model": "mistral",
        "prompt": prompt,
        "stream": False
    })
    
    answer = json.loads(response.text)['response']
    sources = list(set([result.metadata['source'] for result in results]))
    
    return {
        "query": query,
        "answer": answer,
        "sources": sources
    }

if __name__ == "__main__":
    result = ask("What's the process for medical billing denial appeals?")
    print(f"Q: {result['query']}\n")
    print(f"A: {result['answer']}\n")
    print(f"Sources: {', '.join(result['sources'])}")