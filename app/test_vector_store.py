from rag.embedder import Embedder
from rag.vector_store import VectorStore


def main():

    embedder = Embedder()

    chunks = [
        "RAG retrieves relevant information from a knowledge base.",
        "Python is a popular programming language.",
        "Embeddings convert text into numerical vectors.",
        "The Eiffel Tower is located in Paris."
    ]

    embeddings = embedder(chunks)

    dimension = len(embeddings[0])

    store = VectorStore(dimension)

    store.add(embeddings,chunks)

    query = "How does retrieval augmented generation work?"

    query_embedding = embedder.embed([query])[0]

    results = store.search(
        query_embedding,
        top_k=2
    )
    print("\nQuery:")
    print(query)

    print("\nRetrieved chunks:")

    for i, result in enumerate(results):
        print(f"\n{i + 1}. {result}")


if __name__ == "__main__":
    main()


