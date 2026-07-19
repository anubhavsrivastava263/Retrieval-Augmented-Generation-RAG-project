from langchain_community.vectorstores import Chroma
from utils.embeddings import get_embeddings

db = Chroma(
    persist_directory="vectordb",
    embedding_function=get_embeddings()
)

def search(query, k=5):
    results = db.similarity_search(query, k=k)
    return results


if __name__ == "__main__":

    question = input("Ask a question: ")

    docs = search(question)

    print("\nTop Results:\n")

    for i, doc in enumerate(docs, 1):
        print("=" * 60)
        print(f"Result {i}")
        print("=" * 60)
        print(doc.page_content)
        print("\nMetadata:", doc.metadata)
        print()