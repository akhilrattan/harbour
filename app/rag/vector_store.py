from db.database import Database
from pgvector.psycopg import register_vector


class VectorStore:

    def __init__(self):
        self.db = Database()

    def add_document(
        self,
        source: str,
        content: str
    ) -> int:

        with self.db.connect() as connection:

            cursor = connection.execute(
                """
                INSERT INTO documents (source, content)
                VALUES (%s, %s)
                RETURNING id
                """,
                (source, content)
            )

            document_id = cursor.fetchone()[0]

            connection.commit()

        return document_id

    def add_chunks(
        self,
        document_id: int,
        chunks: list[str],
        embeddings: list[list[float]]
    ):

        if len(chunks) != len(embeddings):
            raise ValueError(
                "Chunks and embeddings must have the same length"
            )

        with self.db.connect() as connection:

            register_vector(connection)

            for chunk, embedding in zip(
                chunks,
                embeddings
            ):
                connection.execute(
                    """
                    INSERT INTO chunks
                    (document_id, content, embedding)
                    VALUES (%s, %s, %s)
                    """,
                    (
                        document_id,
                        chunk,
                        embedding
                    )
                )

            connection.commit()

    def search(
        self,
        query_embedding: list[float],
        top_k: int = 3
    ) -> list[str]:

        with self.db.connect() as connection:

            register_vector(connection)

            cursor = connection.execute(
                """
                SELECT content
                FROM chunks
                ORDER BY embedding <=> %s::vector
                LIMIT %s
                """,
                (
                    query_embedding,
                    top_k
                )
            )

            rows = cursor.fetchall()

        return [
            row[0]
            for row in rows
        ]