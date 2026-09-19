import json

from llm.provider import LLMProvider


class Planner:

    def __init__(self, llm: LLMProvider):
        self.llm = llm

    def plan(self, question: str, context : str = "") -> dict:
        
        prompt = f"""
You are the planner for an agentic RAG system.

Decide what the agent should do next.

Available actions:

1. retrieve
   Search the knowledge base.

2. memory
   Search persistent memory.

3. answer
   Stop gathering information and answer.

Previous observations:
{context}

User question:
{question}

Return ONLY valid JSON.

Example:

{{"action": "memory"}}
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