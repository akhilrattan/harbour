from rag.chunker import TextChunker
from rag.embedder import Embedder
from rag.vector_store import VectorStore
from rag.ingestion import IngestionPipeline


def main():

    chunker = TextChunker(
        chunk_size=300,
        overlap=50
    )

    embedder = Embedder()

    vector_store = VectorStore()

    pipeline = IngestionPipeline(
        chunker=chunker,
        embedder=embedder,
        vector_store=vector_store
    )

    result = pipeline.ingest(
        "data/text.txt"
    )

    print("\nIngestion complete!")

    print("Document ID:", result["document_id"])
    print("Source:", result["source"])
    print("Chunks:", result["chunks"])


if __name__ == "__main__":
    main()