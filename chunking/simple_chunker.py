from langchain_text_splitters import RecursiveCharacterTextSplitter
from typing import List


class SimpleChunker:
    def __init__(
        self, chunk_size: int, chunk_overlap: int, separators=["\n\n", "\n", " ", ""]
    ):
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=separators,
        )

    def split_text(self, text: str) -> List[str]:
        chunks = self.splitter.split_text(text)
        return chunks
