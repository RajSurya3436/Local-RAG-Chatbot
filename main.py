from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
from ingest import ingest_document
from rag import get_answer
import shutil, os

app = FastAPI(
    title="RAG Chatbot API",
    description="Upload documents and ask questions about them!"
)

# Pydantic model = defines what the request body looks like
# FastAPI uses this to auto-validate incoming JSON
class ChatRequest(BaseModel):
    question: str

@app.get("/health")
def health():
    """Just to confirm the server is running"""
    return {"status": "ok"}

@app.post("/ingest")
async def ingest(file: UploadFile = File(...)):
    """
    Upload a .pdf or .txt file.
    It gets saved temporarily, indexed into ChromaDB, then deleted.
    """
    allowed = [".pdf", ".txt"]
    ext = os.path.splitext(file.filename)[1].lower()

    if ext not in allowed:
        raise HTTPException(status_code=400, detail="Only .pdf and .txt files allowed")

    # Save uploaded file temporarily
    tmp_path = f"./tmp_{file.filename}"
    with open(tmp_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    # Ingest it
    chunks = ingest_document(tmp_path)

    # Clean up temp file
    os.remove(tmp_path)

    return {"message": f"Successfully ingested '{file.filename}'", "chunks": chunks}

@app.post("/chat")
def chat(req: ChatRequest):
    """
    Ask a question. The API will search your documents
    and use the LLM to generate an answer.
    """
    if not req.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")

    answer = get_answer(req.question)
    return {"question": req.question, "answer": answer}