from utils.loader import load_documents
from utils.chunker import split_documents
from utils.embeddings import get_embeddings

from langchain_community.vectorstores import Chroma

print("Loading documents...")
documents = load_documents()

print(f"Loaded {len(documents)} pages.")

print("Splitting into chunks...")
chunks = split_documents(documents)

print(f"Created {len(chunks)} chunks.")

print("Creating Chroma database...")

vectordb = Chroma.from_documents(
    documents=chunks,
    embedding=get_embeddings(),
    persist_directory="vectordb"
)

print("Documents indexed successfully!")