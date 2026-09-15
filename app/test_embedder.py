from rag.embedder import Embedder


def main():

    embedder = Embedder()

    texts = [
        "RAG retrieves relevant information",
        "Retrieval augmented generation uses external knowledge",
        "The weather is sunny today"
    ]

    embeddings = embedder.embed(texts)

    print("Number of embeddings:", len(embeddings))

    print("Embedding dimension:", len(embeddings[0]))

    print("\nFirst embedding:")
    print(embeddings[0][:10])


if __name__ == "__main__":
    main()