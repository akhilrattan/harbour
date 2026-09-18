from rag.embedder import Embedder
from rag.vector_store import VectorStore


class Retriever:

    def __init__(
        self,
        embedder: Embedder,
        vector_store: VectorStore
    ):
        self.embedder = embedder
        self.vector_store = vector_store

    def retrieve(
        self,
        query: str,
        top_k: int = 3
    ) -> list[str]:

        query_embedding = self.embedder.embed(
            [query]
        )[0]

        return self.vector_store.search(
            query_embedding,
            top_k
        )