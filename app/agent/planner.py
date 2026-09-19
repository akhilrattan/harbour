import json

from llm.provider import LLMProvider


class Planner:

    def __init__(self, llm: LLMProvider):
        self.llm = llm

    def plan(
    self,
    question: str,
    action_history: list[str],
    context: str
) -> dict:
        
        prompt = f"""
You are the planner for an agentic RAG system.

Your job is to decide what the agent should do NEXT.

Available actions:

1. retrieve
   Search the knowledge base for relevant information.

2. memory
   Search persistent memory for information about the
   user or previous decisions.

3. answer
   Stop gathering information and produce the final answer.

Important rules:

- Do not repeat an action unnecessarily.
- If useful information has already been collected,
  you may choose answer.
- If the question requires both user-specific memory
  and knowledge-base information, use both.
- Choose answer when enough information has been collected.

Previous actions:
{action_history}

Information collected so far:
{context}

User question:
{question}

Return ONLY valid JSON.

Example:

{{"action": "memory"}}

or:

{{"action": "retrieve"}}

or:

{{"action": "answer"}}
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