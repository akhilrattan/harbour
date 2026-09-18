from rag.embedder import Embedder
from rag.vector_store import VectorStore
from rag.retriever import Retriever


def main():

    embedder = Embedder()

    vector_store = VectorStore()

    retriever = Retriever(
        embedder,
        vector_store
    )

    query = "What is retrieval augmented generation?"

    results = retriever.retrieve(
        query,
        top_k=3
    )

    print("\nQuery:")
    print(query)

    print("\nRetrieved documents:")

    for i, result in enumerate(results, start=1):

        print(f"\n{i}.")
        print(result)


if __name__ == "__main__":
    main()