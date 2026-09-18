from db.database import Database
from pgvector.psycopg import register_vector


class VectorStore:

    def __init__(self):
        self.db = Database()

    def add_documents (self, source : str, content : str) -> int :

        with self.db.connect() as connection:
            
            cursor = connection.execute(
                """
                INSERT INTO documents(source,content)
                VALUES(%s,%s)
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
            query_embeddings:list[str],
            top_k = int,
       ) -> list[str]:
        
        with self.db.connect as connection :
            register_vector(connection)

            cursor = connection.execute(
                """
                SELECT content
                FROM chunks 
                ORDER BY embeddings <=> %s
                LIMIT %s
                """,
                (
                    query_embeddings,
                    top_k
                )
            )
            rows = cursor.fetchall()
        return [
            rows[0]
            for row in rows
        ]