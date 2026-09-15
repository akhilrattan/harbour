from llm.provider import OllamaProvider
from rag.embedder import Embedder
from rag.vector_store import VectorStore
from rag.retriever import Retriever
from rag.rag import RAG


def main():

    # -------------------------
    # 1. Create our components
    # -------------------------

    embedder = Embedder()

    llm = OllamaProvider(
        "qwen2.5:3b"
    )

    # -------------------------
    # 2. Create knowledge base
    # -------------------------

    chunks = [
        "RAG stands for Retrieval Augmented Generation.",
        "RAG allows a language model to retrieve information "
        "from an external knowledge base before generating an answer.",
        "Embeddings represent text as numerical vectors.",
        "FAISS can be used to perform similarity search over vectors.",
        "Python is a popular programming language."
    ]

    # -------------------------
    # 3. Embed the chunks
    # -------------------------

    embeddings = embedder.embed(chunks)

    dimension = len(embeddings[0])

    # -------------------------
    # 4. Create vector store
    # -------------------------

    vector_store = VectorStore(dimension)

    vector_store.add(
        embeddings,
        chunks
    )

    # -------------------------
    # 5. Create retriever
    # -------------------------

    retriever = Retriever(
        embedder,
        vector_store
    )

    # -------------------------
    # 6. Create RAG system
    # -------------------------

    rag = RAG(
        llm,
        retriever
    )

    # -------------------------
    # 7. Ask a question
    # -------------------------

    question = "What does RAG stand for?"

    answer = rag.answer(question)

    print("\nQuestion:")
    print(question)

    print("\nAnswer:")
    print(answer)


if __name__ == "__main__":
    main()