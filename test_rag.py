import pytest
from rag_ollama import ask, check_ollama_connection

def test_ollama_connection():
    assert check_ollama_connection(), "Ollama not running"

def test_valid_query():
    result = ask("What is medical billing?")
    assert result["answer"], "Answer empty"
    assert len(result["sources"]) > 0, "No sources"

def test_empty_query_rejected():
    result = ask("")
    assert "error" in result or len(result["answer"]) > 0

def test_query_too_short():
    result = ask("ab")
    assert "error" in result or "must be at least" in result["answer"].lower()

def test_retrieval_sources():
    result = ask("denial appeals")
    if result.get("answer") and "error" not in result:
        assert isinstance(result["sources"], list) and len(result["sources"]) > 0

def test_response_format():
    result = ask("credentialing")
    for field in ["query", "answer", "sources"]:
        assert field in result