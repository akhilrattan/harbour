from llm.provider import LLMProvider

from rag.retriever import Retriever
from memory.memory import Memory

from agent.planner import Planner
from agent.state import AgentState


class Agent:

    def __init__(
        self,
        llm: LLMProvider,
        retriever: Retriever,
        memory: Memory
    ):

        self.llm = llm
        self.retriever = retriever
        self.memory = memory

        self.planner = Planner(llm)

    def run(
        self,
        question: str
    ) -> AgentState:

        state = AgentState(
            question=question
        )

        # -------------------------
        # 1. PLAN
        # -------------------------

        decision = self.planner.plan(
            question
        )

        state.action = decision["action"]

        # -------------------------
        # 2. RETRIEVE
        # -------------------------

        if state.action == "retrieve":

            state.retrieved_chunks = (
                self.retriever.retrieve(
                    question,
                    top_k=3
                )
            )

            state.context = (
                "\n\n".join(
                    state.retrieved_chunks
                )
            )

        # -------------------------
        # 3. MEMORY
        # -------------------------

        elif state.action == "memory":

            state.memories = (
                self.memory.search(
                    question,
                    top_k=3
                )
            )

            state.context = (
                "\n\n".join(
                    state.memories
                )
            )

        # -------------------------
        # 4. BUILD PROMPT
        # -------------------------

        if state.action == "answer":

            prompt = f"""
Answer the user's question directly.

Question:
{question}
"""

        else:

            prompt = f"""
Answer the user's question using the
information below.

If the information is insufficient,
say that you don't know.

Information:
{state.context}

Question:
{question}
"""

        # -------------------------
        # 5. GENERATE ANSWER
        # -------------------------

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