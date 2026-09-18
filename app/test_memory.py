from rag.embedder import Embedder
from memory.memory import Memory


def main():

    embedder = Embedder()

    memory = Memory(
        embedder=embedder
    )

    # Store memories
    memory.save(
        "The user is building an agentic RAG system."
    )

    memory.save(
        "The user wants to deploy the entire project online."
    )

    memory.save(
        "The project uses PostgreSQL and pgvector."
    )

    # Search memory
    query = "What database is being used?"

    results = memory.search(
        query,
        top_k=2
    )

    print("\nQuery:")
    print(query)

    print("\nRelevant memories:")

    for i, result in enumerate(
        results,
        start=1
    ):
        print(f"\n{i}. {result}")


if __name__ == "__main__":
    main()