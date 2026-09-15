from llm.provider import OllamaProvider
from rag.embedder import Embedder
from rag.vector_store import VectorStore
from rag.retriever import Retriever
from agent.agent import Agent


def main():

    # LLM
    llm = OllamaProvider("qwen2.5:3b")

    # Embedding model
    embedder = Embedder()

    # Knowledge base
    chunks = [
        "RAG stands for Retrieval Augmented Generation.",
        "RAG allows an LLM to retrieve information "
        "from an external knowledge base.",
        "FAISS is a library for efficient similarity search "
        "over vectors.",
        "Embeddings represent text as numerical vectors."
    ]

    # Create embeddings
    embeddings = embedder.embed(chunks)

    # Create vector store
    dimension = len(embeddings[0])

    vector_store = VectorStore(dimension)

    vector_store.add(
        embeddings,
        chunks
    )

    # Retriever
    retriever = Retriever(
        embedder,
        vector_store
    )

    # Agent
    agent = Agent(
        llm,
        retriever
    )

    # Questions
    questions = [
        "What does RAG stand for?",
        "Say hello to me."
    ]

    for question in questions:

        print("\n====================")

        print("Question:")
        print(question)

        answer = agent.run(question)

        print("\nAnswer:")
        print(answer)


if __name__ == "__main__":
    main()