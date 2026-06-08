import ollama
from sentence_transformers import SentenceTransformer
from vectorstore import collection

embedder = SentenceTransformer("all-MiniLM-L6-v2")

def get_answer(question: str) -> str:
  
    question_vector = embedder.encode(question).tolist()

    
    results = collection.query(
        query_embeddings=[question_vector],
        n_results=3  
    )

   
    context_chunks = results["documents"][0]
    context = "\n\n".join(context_chunks)
    prompt = f"""You are a helpful assistant. Answer the question using ONLY the context below.
If the answer is not in the context, say "I don't have enough information to answer that."

Context:
{context}

Question: {question}

Answer:"""
 response = ollama.chat(
        model="llama3.2:3b",
        messages=[{"role": "user", "content": prompt}]
    )

    return response["message"]["content"]
