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
        question: str,
        max_steps: int = 5
    ) -> AgentState:

        state = AgentState(
            question=question
        )

        for step in range(max_steps):

            print(f"\n--- Agent Step {step + 1} ---")

            # 1. PLAN
            decision = self.planner.plan(
                question,
                state.context 
            )

            action = decision["action"]

            state.action = action

            print("Action:", action)

            # 2. ACT
            if action == "retrieve":

                chunks = self.retriever.retrieve(
                    question,
                    top_k=3
                )

                state.retrieved_chunks.extend(
                    chunks
                )

                state.context = (
                    "\n\n".join(
                        state.retrieved_chunks
                    )
                )

                print("Retrieved:", len(chunks))

            elif action == "memory":

                memories = self.memory.search(
                    question,
                    top_k=3
                )

                state.memories.extend(
                    memories
                )

                state.context = (
                    "\n\n".join(
                        state.memories
                    )
                )

                print("Memories:", len(memories))

            elif action == "answer":

                break

        # 3. GENERATE FINAL ANSWER

        if state.context:

            prompt = f"""
Answer the user's question using the
information collected by the agent.

If the information is insufficient,
say that you don't know.

Collected information:
{state.context}

Question:
{question}
"""

        else:

            prompt = f"""
Answer the user's question directly.

Question:
{question}
"""

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