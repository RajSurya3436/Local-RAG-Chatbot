import chromadb

# Create a ChromaDB client that saves data to disk (persistent)
client = chromadb.PersistentClient(path="./chroma_db")

# A "collection" is like a table in a normal database
# It holds all our document chunks + their vectors
collection = client.get_or_create_collection(name="documents")