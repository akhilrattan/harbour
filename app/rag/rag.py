from llm.provider import LLMProvider
from rag.retriever import Retriever


class RAG:

    def __init__(
        self,
        llm: LLMProvider,
        retriever: Retriever
    ):
        self.llm = llm
        self.retriever = retriever

    def answer(
        self,
        question: str,
        top_k: int = 3
    ) -> str:
        chunks = self.retriever.retrieve(
            question,
            top_k=top_k
        )
        context = "\n\n".join(chunks)
        prompt = f"""
You are a helpful assistant.

Answer the user's question using ONLY the
information provided in the context.

If the answer cannot be found in the context,
say that you don't have enough information.

Context:
{context}

Question:
{question}
"""
        messages = [
            {
                "role" : "user",
                "content": "prompt"
            }
        ]

        response = self.llm.chat(messages)
        return response

