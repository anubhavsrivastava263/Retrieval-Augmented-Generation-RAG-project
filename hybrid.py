from rank_bm25 import BM25Okapi

class BM25Retriever:

    def __init__(self, docs):

        self.docs = docs

        tokenized_docs = [
            doc.page_content.split()
            for doc in docs
        ]

        self.bm25 = BM25Okapi(tokenized_docs)

    def search(self, query, k=5):

        scores = self.bm25.get_scores(query.split())

        ranked = sorted(
            zip(scores, self.docs),
            key=lambda x: x[0],
            reverse=True
        )

        return [doc for score, doc in ranked[:k]]