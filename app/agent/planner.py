import json

from llm.provider import LLMProvider


class Planner:

    def __init__(self, llm: LLMProvider):
        self.llm = llm

    def plan(self, question: str) -> dict:

        prompt = f"""
You are the planner for an agentic RAG system.

Decide what the agent should do with the user's question.

Available actions:

1. retrieve
   Use this when the answer should come from
   the user's knowledge base.

2. memory
   Use this when the answer may depend on
   information remembered about the user
   or previous conversations.

3. answer
   Use this when no retrieval or memory is needed.

Return ONLY valid JSON.

Examples:

Question:
"What does RAG stand for?"

{{"action": "retrieve"}}

Question:
"What database did I choose for this project?"

{{"action": "memory"}}

Question:
"Hello!"

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
            "memory"
            "answer"
        }:
            raise ValueError(
                f"Unknown planner action: {decision}"
            )

        return decision