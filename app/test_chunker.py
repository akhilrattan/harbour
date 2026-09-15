from rag.chunker import TextChunker


def main():

    text = """
    RAG stands for Retrieval Augmented Generation.
    RAG allows an LLM to retrieve relevant information
    from an external knowledge base before generating an answer.
    Instead of relying only on the knowledge stored in the model,
    RAG provides additional context to the model.
    """

    chunker = TextChunker(chunk_size=100, overlap=50)

    chunks = chunker.split(text)
    
    for i, chunk in enumerate(chunks):
        print(f"\n--- Chunk {i + 1} ---")
        print(chunk)


if __name__ == "__main__":
    main()