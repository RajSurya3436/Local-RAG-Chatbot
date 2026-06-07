import pdfplumber
from sentence_transformers import SentenceTransformer
from vectorstore import collection
import uuid

# Load a small, fast embedding model
# "all-MiniLM-L6-v2" is only ~80MB and works great on 8GB RAM
# It converts text → a list of 384 numbers (a vector)
embedder = SentenceTransformer("all-MiniLM-L6-v2")

def read_file(filepath: str) -> str:
    """Read text from a .txt or .pdf file"""
    if filepath.endswith(".pdf"):
        text = ""
        with pdfplumber.open(filepath) as pdf:
            for page in pdf.pages:
                text += page.extract_text() or ""
        return text
    else:
        with open(filepath, "r", encoding="utf-8") as f:
            return f.read()

def chunk_text(text: str, chunk_size=500, overlap=50) -> list[str]:
    """
    Split text into overlapping chunks.
    
    Why overlap? Imagine a sentence split across two chunks:
    Chunk 1: "...the answer is"
    Chunk 2: "42, which means..."
    Without overlap, neither chunk makes sense alone.
    With overlap, each chunk carries a bit of the previous one.
    """
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - overlap  # step back by overlap amount
    return chunks

def ingest_document(filepath: str):
    """Full pipeline: read → chunk → embed → store"""
    print(f" Reading file: {filepath}")
    text = read_file(filepath)

    print(f" Splitting into chunks...")
    chunks = chunk_text(text)

    print(f" Converting {len(chunks)} chunks to vectors...")
    embeddings = embedder.encode(chunks).tolist()

    print(f" Saving to ChromaDB...")
    collection.add(
        documents=chunks,
        embeddings=embeddings,
        ids=[str(uuid.uuid4()) for _ in chunks]  # unique ID for each chunk
    )
    print(f" Done! Ingested {len(chunks)} chunks.")
    return len(chunks)
