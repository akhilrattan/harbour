from llm.provider import OllamaProvider


def main():

    llm = OllamaProvider("qwen2.5:3b")

    messages = [
        {
            "role": "user",
            "content": "Explain RAG in two sentences."
        }
    ]

    response = llm.chat(messages)

    print(response)


if __name__ == "__main__":
    main()