# RAG System: Medical Billing Knowledge Base

A retrieval-augmented generation system built over 264 medical billing articles. Demonstrates embeddings, vector search, semantic retrieval, and LLM integration.

## What It Does

Retrieves relevant passages from a medical billing knowledge base and generates grounded answers via LLM. Eliminates hallucinations by enforcing retrieval-based grounding.

**Example:**
```
Q: What's the process for medical billing denial appeals?

A: 1. Denial Management and AR Follow-Up: Routes claims through appropriate channels for repair, resubmission, or challenge within time limits.
   2. Correction: Claim is corrected based on denial reasons (uncovered services, expired auth, medical necessity disputes, coordination of benefits issues).
   3. Resubmission: Corrected claim resubmitted for re-adjudication.
   4. Appeal: If denied, appealed per payer guidelines.
   5. Review and Response: Payer reviews and responds. Additional info may be requested.
   6. Final Decision: Based on appeal review. Further appeals possible per payer policies.

Sources: medical_billing_in_washington.txt, how_claims_are_processed_in_medical_billing.txt, best_medical_billing_solutions_in_the_usa.txt
```

## Architecture

**Data Pipeline:**
- 326 articles scraped from credexhealthcare.com via readability parser (handles JavaScript-rendered content)
- 264 articles passed content quality threshold (>300 chars)
- Recursive character splitting: 800-char chunks with 150-char overlap
- Total: 5,533 chunks

**Embeddings & Retrieval:**
- Model: `sentence-transformers/all-MiniLM-L6-v2` (22MB, runs locally)
- Vector DB: Chroma (persistent storage in `/chroma_db`)
- Retrieval: Cosine similarity search, top-4 chunks per query

**Generation:**
- Local: Mistral 7B via Ollama (no API key, runs offline, ~4GB model)
- Alternative: Claude API (requires billing)

## Why These Choices

**Readability parser over DOM selectors:** Elementor pages use JavaScript rendering. Readability (Mozilla algorithm) extracts content regardless of DOM structure—more reliable than targeting specific classes.

**Sentence-transformers over BERT:** Pre-trained on semantic similarity. Fast inference, small footprint, domain-agnostic (works across healthcare, finance, technical content).

**Chroma over Pinecone:** Persistent local storage, zero infrastructure overhead, sufficient for 5K+ chunks. Pinecone adds latency for small-scale projects.

**Ollama for local deployment:** Eliminates API dependency, reduces latency, enables offline operation. Trade-off: larger model file, requires local compute. For cloud deployment, swap to Hugging Face Inference API or Claude.

## Setup

### Local Execution

**Requirements:**
- Python 3.10+
- 8GB+ RAM (for Ollama)
- Ollama installed (https://ollama.ai)

**Steps:**

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Pull Mistral model (one-time, ~4GB)
ollama pull mistral

# 3. Ollama runs in background automatically

# 4. Test retrieval + generation
python rag_ollama.py
```

Runs offline. Expects `chroma_db/` (pre-built, included in repo).

### Files

- `rag_ollama.py` — Main RAG pipeline (retrieval + Mistral generation)
- `ingest.py` — Document chunking + embedding pipeline (already run)
- `api.py` — FastAPI wrapper (requires Claude or HF API for deployment)
- `scraper_v2.py` — Web scraper for Credex articles (reference only)
- `chroma_db/` — Vector database (gitignored, rebuilt if needed)
- `requirements.txt` — Dependencies

## Performance

**Latency (local Ollama):**
- Retrieval: ~200ms (5,533 chunks)
- Generation: 5-10s (Mistral 7B on CPU; faster on GPU)
- Total: ~6-11s end-to-end

**Quality:**
- Retrieval: Top-4 chunks are semantically relevant (manual verification on 10+ queries)
- Generation: Answers cite sources, no hallucinations observed in tested queries

## Cloud Deployment (Not Implemented)

To deploy `api.py` to Render/Heroku:
1. Replace Ollama calls with Hugging Face Inference API or Claude SDK
2. Set `HUGGINGFACE_API_KEY` or `ANTHROPIC_API_KEY` env var
3. Update `api.py` to call cloud endpoint instead of localhost:11434
4. Deploy with `Procfile` and `requirements.txt`

**Why not deployed:** Ollama runs locally; cloud containers can't easily run 4GB models. API-based alternatives require payment or rate limits. Local demo demonstrates architecture understanding better than a constrained cloud version.

## Technical Insights

**Chunk size matters:** 800 chars balances context (medical billing concepts span multiple sentences) with precision (avoids distant-document relevance).

**Overlap strategy:** 150-char overlap preserves cross-chunk context. Medical billing has domain-specific terminology that benefits from sentence-level redundancy.

**Readability parsing:** Robustness over specificity. A selector-based parser broke on 62/326 articles. Readability handled all.

**Embedding model selection:** MiniLM trades accuracy (~2% behind larger models) for speed and size. Acceptable trade-off for retrieval; generation handles semantic nuance.

## What This Demonstrates

- **Vector embeddings:** Semantic similarity without keyword matching
- **Vector databases:** Efficient storage and search at scale (5K+ chunks)
- **Retrieval-augmented generation:** Grounding LLM outputs in source data
- **Data pipelines:** Web scraping, parsing, chunking, embedding
- **Local LLM deployment:** Running inference without cloud dependencies
- **Trade-offs:** Local vs. cloud, accuracy vs. speed, scope vs. complexity

## Next Steps

- Fine-tune embeddings on medical billing terminology (domain-specific model)
- Add multi-turn conversation with context carryover
- Implement evaluation metrics (NDCG, retrieval precision@k)
- Deploy to cloud with API-based LLM

## License

Private—for portfolio demonstration.
