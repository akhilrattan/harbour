from db.database import Database
from pgvector.psycopg import register_vector

from rag.embedder import Embedder


class Memory:

    def __init__(self, embedder: Embedder):

        self.db = Database()
        self.embedder = embedder

    def save(self, content: str):

        embedding = self.embedder.embed(
            [content]
        )[0]

        with self.db.connect() as connection:

            register_vector(connection)

            connection.execute(
                """
                INSERT INTO memories
                (content, embedding)
                VALUES (%s, %s)
                """,
                (
                    content,
                    embedding
                )
            )

            connection.commit()

    def search(
        self,
        query: str,
        top_k: int = 3
    ) -> list[str]:

        query_embedding = self.embedder.embed(
            [query]
        )[0]

        with self.db.connect() as connection:

            register_vector(connection)

            cursor = connection.execute(
                """
                SELECT content
                FROM memories
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

    def get_all(self) -> list[str]:

        with self.db.connect() as connection:

            cursor = connection.execute(
                """
                SELECT content
                FROM memories
                ORDER BY id
                """
            )

            rows = cursor.fetchall()

        return [
            row[0]
            for row in rows
        ]