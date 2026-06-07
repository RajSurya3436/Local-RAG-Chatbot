import ollama
from sentence_transformers import SentenceTransformer
from vectorstore import collection

embedder = SentenceTransformer("all-MiniLM-L6-v2")

def get_answer(question: str) -> str:
    # Step 1: Convert question to a vector
    question_vector = embedder.encode(question).tolist()

    # Step 2: Find top 3 most similar chunks in ChromaDB
    results = collection.query(
        query_embeddings=[question_vector],
        n_results=3  # get top 3 relevant chunks
    )

    # results["documents"] looks like [["chunk1", "chunk2", "chunk3"]]
    # We join them into one context block
    context_chunks = results["documents"][0]
    context = "\n\n".join(context_chunks)

    # Step 3: Build the prompt
    # We tell the model: "Only answer from this context, don't make things up"
    prompt = f"""You are a helpful assistant. Answer the question using ONLY the context below.
If the answer is not in the context, say "I don't have enough information to answer that."

Context:
{context}

Question: {question}

Answer:"""

    # Step 4: Send to Ollama (running locally on your machine)
    # llama3.2:3b is a 3 billion parameter model — small enough for 8GB RAM
    response = ollama.chat(
        model="llama3.2:3b",
        messages=[{"role": "user", "content": prompt}]
    )

    return response["message"]["content"]
