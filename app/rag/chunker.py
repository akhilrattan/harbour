class TextChunker:

    def __init__(self, chunk_size: int = 500, overlap: int = 50):
        self.chunk_size = chunk_size
        self.overlap = overlap

    def split(self, text: str) -> list[str]:
        if self.chunk_size <= 0:
            raise ValueError("chunk_size must be greater than 0")

        if self.overlap < 0:
            raise ValueError("overlap cannot be negative")

        if self.overlap >= self.chunk_size:
            raise ValueError("overlap must be smaller than chunk_size")

        start = 0

        chunks = []

        while start < len(text):
            end = start + self.chunk_size
            chunk = text[start:end]
            chunks.append(chunk)
            start = end - self.overlap
        
        return chunks
