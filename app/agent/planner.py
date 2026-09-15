import json

from llm.provider import LLMProvider


class Planner:

    def __init__(self, llm: LLMProvider):
        self.llm = llm

    def plan(self, question: str) -> dict:

        prompt = f"""
You are a planner for a RAG agent.

Decide whether the user's question requires
searching the knowledge base.

Return ONLY valid JSON.

If the knowledge base is needed:
{{"action": "retrieve"}}

If the question can be answered without the
knowledge base:
{{"action": "answer"}}

User question:
{question}
"""

        messages = [
            {
                "role": "user",
                "content": prompt
            }
        ]

        response = self.llm.chat(messages)

        try:
            decision = json.loads(response)
        except json.JSONDecodeError:
            raise ValueError(
                f"Planner returned invalid JSON: {response}"
            )
        
        if decision.get("action") not in {
            "retrieve",
            "answer"
        }:
            raise ValueError(
                f"Unknown planner action: {decision}"
            )

        return decision