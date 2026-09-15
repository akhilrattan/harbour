from llm.provider import LLMProvider
from rag.retriever import Retriever
from agent.planner import Planner
from agent.state import AgentState


class Agent:

    def __init__(
        self,
        llm: LLMProvider,
        retriever: Retriever
    ):
        self.llm = llm
        self.retriever = retriever
        self.planner = Planner(llm)

    def run(self, question: str) -> AgentState:

        state = AgentState(
            question=question
        )
        # Step 1: Ask the planner what to do
        decision = self.planner.plan(state.question)

        state.action = decision["action"]

        # Step 2: If retrieval is needed
        if state.action == "retrieve":

            state.chunks = self.retriever.retrieve(
                state.question,
                top_k=3
            )

            state.context = "\n\n".join(state.chunks)

            prompt = f"""
Answer the user's question using the context below.

If the context does not contain enough information,
say that you don't know.

Context:
{state.context}

Question:
{state.question}
"""

        else:

            prompt = f"""
Answer the following question directly.

Question:
{state.question}
"""

        # Step 3: Ask the LLM
        state.messages = [
            {
                "role": "user",
                "content": prompt
            }
        ]

        state.answer = self.llm.chat(
            state.messages
        )
        return state