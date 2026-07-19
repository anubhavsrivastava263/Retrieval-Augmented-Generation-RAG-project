from utils.loader import load_documents
from utils.chunker import split_documents

from retrieve import search
from hybrid import BM25Retriever

print("Loading documents...")

docs = load_documents()

chunks = split_documents(docs)

bm25 = BM25Retriever(chunks)

query = input("Ask a question: ")

print("\nVECTOR SEARCH\n")

vector_results = search(query)

for i, doc in enumerate(vector_results, 1):
    print("="*60)
    print(f"Vector {i}")
    print(doc.page_content[:400])

print("\nBM25 SEARCH\n")

bm25_results = bm25.search(query)

for i, doc in enumerate(bm25_results, 1):
    print("="*60)
    print(f"BM25 {i}")
    print(doc.page_content[:400])