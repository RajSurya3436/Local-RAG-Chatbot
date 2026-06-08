import pdfplumber
from sentence_transformers import SentenceTransformer
from vectorstore import collection
import uuid


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
    
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - overlap  # step back by overlap amount
    return chunks

def ingest_document(filepath: str):
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
        ids=[str(uuid.uuid4()) for _ in chunks] 
    )
    print(f" Done! Ingested {len(chunks)} chunks.")
    return len(chunks)
