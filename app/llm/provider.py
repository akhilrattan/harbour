from abc import ABC, abstractmethod
import requests


class LLMProvider(ABC):
    """
    Common interface for all LLM providers.
    """

    @abstractmethod
    def chat(self, messages: list[dict]) -> str:
        """
        Send messages to an LLM and return its response as text.
        """
        pass


class OllamaProvider(LLMProvider):

    def __init__(self, model: str, base_url: str = "http://localhost:11434"):
        self.model = model
        self.base_url = base_url

    def chat(self, messages: list[dict]) -> str:

        response = requests.post(
            f"{self.base_url}/api/chat",
            json={
                "model": self.model,
                "messages": messages,
                "stream": False,
            },
        )

        response.raise_for_status()

        data = response.json()

        return data["message"]["content"]