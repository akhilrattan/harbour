from db.database import Database
from pgvector.psycopg import register_vector


class VectorStore:

    def __init__(self, dimension: int):
        self.dimension = dimension

        # FAISS index for Euclidean distance
        self.index = faiss.IndexFlatL2(dimension)

        # Keep the original chunks alongside the vectors
        self.chunks = []

    def add(self, embeddings: list[list[float]], chunks: list[str]):
        if len(embeddings) != len(chunks):
            raise ValueError(
                "Number of embeddings must match number of chunks"
            )
        vectors = np.array(
            embeddings,
            dtype=np.float32
        )
        self.index.add(vectors)
        self.chunks.extend(chunks)


    def search(
        self,
        query_embedding: list[float],
        top_k: int = 3
        ) -> list[str]:
        query_vector = np.array(
            [query_embedding],
            dtype=np.float32
        )

        distances, indices = self.index.search(
            query_vector,
            top_k
        )
        results = []

        for index in indices[0]:

            if index == -1:
                continue

            results.append(self.chunks[index])

        return results