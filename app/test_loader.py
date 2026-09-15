from rag.loader import TextLoader


def main():
    # Create the loader
    loader = TextLoader("data/text.txt")
    # Load the document
    text = loader.load()
    # Print the text
    print(text)
    pass


if __name__ == "__main__":
    main()