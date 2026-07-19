from retrieve import search
from utils.reranker import rerank
from generator import generate_answer

query = input("Ask a question: ")

docs = search(query)
docs = rerank(query, docs)

context = "\n\n".join(doc.page_content for doc in docs[:3])

answer = generate_answer(query, context)

print("\n" + "=" * 70)
print("ANSWER")
print("=" * 70)
print(answer)

print("\nSOURCES")
print("=" * 70)

for doc in docs[:3]:
    print(
        f"{doc.metadata.get('source')} "
        f"(Page {doc.metadata.get('page', 'Unknown') + 1})"
    )