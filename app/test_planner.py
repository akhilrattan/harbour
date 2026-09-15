from llm.provider import OllamaProvider
from agent.planner import Planner


def main():

    llm = OllamaProvider("qwen2.5:3b")

    planner = Planner(llm)

    questions = [
        "What does RAG stand for?",
        "Hello, how are you?",
        "According to the documents, what is FAISS?"
    ]

    for question in questions:

        decision = planner.plan(question)

        print("\nQuestion:")
        print(question)

        print("Decision:")
        print(decision)


if __name__ == "__main__":
    main()