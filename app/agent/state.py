from dataclasses import dataclass, field


@dataclass
class AgentState:

    question: str

    action: str | None = None

    retrieved_chunks: list[str] = field(default_factory=list)

    context: str = ""

    answer: str = ""

    messages: list[dict] = field(default_factory=list)