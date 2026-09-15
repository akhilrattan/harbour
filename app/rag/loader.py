from pathlib import Path


class TextLoader:

    def __init__(self, file_path: str):
        self.file_path = Path(file_path)

    def load(self) -> str:
        if not self.file_path.exists():
            raise FileNotFoundError(
                f"File not found: {self.file_path}"
            )

        if not self.file_path.is_file():
            raise ValueError(
                f"Path is not a file: {self.file_path}"
            )

        return self.file_path.read_text(encoding="utf-8")
