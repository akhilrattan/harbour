from llm.provider import LLMProvider
from rag.retriever import Retriever
from agent.planner import Planner


class Agent:

    def __init__(
        self,
        llm: LLMProvider,
        retriever: Retriever
    ):
        self.llm = llm
        self.retriever = retriever
        self.planner = Planner(llm)

    def run(self, question: str) -> str:

        # Step 1: Ask the planner what to do
        decision = self.planner.plan(question)

        # Step 2: If retrieval is needed
        if decision["action"] == "retrieve":

            chunks = self.retriever.retrieve(
                question,
                top_k=3
            )

            context = "\n\n".join(chunks)

            prompt = f"""
Answer the user's question using the context below.

If the context does not contain enough information,
say that you don't know.

Context:
{context}

Question:
{question}
"""

        else:

            prompt = f"""
Answer the following question directly.

Question:
{question}
"""

        # Step 3: Ask the LLM
        messages = [
            {
                "role": "user",
                "content": prompt
            }
        ]

        return self.llm.chat(messages)