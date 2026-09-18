from pathlib import Path

from rag.loader import TextLoader
from rag.chunker import TextChunker
from rag.embedder import Embedder
from rag.vector_store import VectorStore


class IngestionPipeline:

    def __init__(
        self,
        chunker: TextChunker,
        embedder: Embedder,
        vector_store: VectorStore
    ):
        self.chunker = chunker
        self.embedder = embedder
        self.vector_store = vector_store

    def ingest(self, file_path: str):

        # 1. Load document
        loader = TextLoader(file_path)

        text = loader.load()

        # 2. Split into chunks
        chunks = self.chunker.split(text)

        # 3. Generate embeddings
        embeddings = self.embedder.embed(chunks)

        # 4. Store document
        document_id = self.vector_store.add_documents(
            source=Path(file_path).name,
            content=text
        )

        # 5. Store chunks + embeddings
        self.vector_store.add_chunks(
            document_id=document_id,
            chunks=chunks,
            embeddings=embeddings
        )

        return {
            "document_id": document_id,
            "source": Path(file_path).name,
            "chunks": len(chunks)
        }