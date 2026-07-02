from pathlib import Path
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

print("🔧 Initializing embeddings...")
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

print("📂 Loading documents...")
docs_path = Path("docs")
documents = []

for file in docs_path.glob("*.txt"):
    with open(file, "r", encoding="utf-8") as f:
        content = f.read()
        documents.append({
            "content": content,
            "source": file.name
        })

print(f"   Loaded {len(documents)} files")

print("✂️  Chunking documents...")
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=800,
    chunk_overlap=150,
    separators=["\n\n", "\n", ". ", " ", ""]
)

chunked_docs = []
for doc in documents:
    chunks = text_splitter.split_text(doc["content"])
    for chunk in chunks:
        chunked_docs.append({
            "text": chunk,
            "source": doc["source"]
        })

print(f"   Created {len(chunked_docs)} chunks")

print("🔍 Embedding chunks...")
db = Chroma.from_texts(
    texts=[doc["text"] for doc in chunked_docs],
    metadatas=[{"source": doc["source"]} for doc in chunked_docs],
    embedding=embeddings,
    persist_directory="./chroma_db"
)

print(f"\n✅ Vector database created: ./chroma_db")
print(f"   Total chunks: {len(chunked_docs)}")
print(f"   Model: all-MiniLM-L6-v2")