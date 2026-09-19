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

            print(
                f"\n--- Agent Step {step + 1} ---"
            )

            # -------------------------
            # 1. PLAN
            # -------------------------

            decision = self.planner.plan(
                question=question,
                action_history=state.action_history,
                context=state.context
            )

            action = decision["action"]

            state.action = action

            state.action_history.append(
                action
            )

            print("Action:", action)

            # -------------------------
            # 2. ANSWER
            # -------------------------

            if action == "answer":
                break

            # -------------------------
            # 3. MEMORY
            # -------------------------

            if action == "memory":

                memories = self.memory.search(
                    question,
                    top_k=3
                )

                state.memories.extend(
                    memories
                )

                observation = (
                    "Memory search returned:\n"
                    + "\n".join(memories)
                )

            # -------------------------
            # 4. RETRIEVE
            # -------------------------

            elif action == "retrieve":

                chunks = self.retriever.retrieve(
                    question,
                    top_k=3
                )

                state.retrieved_chunks.extend(
                    chunks
                )

                observation = (
                    "Knowledge-base search returned:\n"
                    + "\n".join(chunks)
                )

            else:

                raise ValueError(
                    f"Unknown action: {action}"
                )

            # -------------------------
            # 5. OBSERVE
            # -------------------------

            state.observations.append(
                observation
            )

            state.context = "\n\n".join(
                state.observations
            )

            print(
                "Observation:",
                observation[:300]
            )

        # -------------------------
        # 6. FINAL ANSWER
        # -------------------------

        prompt = f"""
You are the final answer generator.

Answer the user's question using the
information collected by the agent.

If the collected information is insufficient,
say that you don't know.

Collected information:

{state.context}

User question:

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