import time

import chunker as rust_chunker
import pytest
import tiktoken


def chunk_text(
    encoding: tiktoken.Encoding, text: str, max_chunk_size: int, overlap: int
) -> tuple[list[str], list[int], int]:
    tokens = encoding.encode(text)
    total_tokens = len(tokens)
    if total_tokens <= max_chunk_size:
        return [text], [tokens], total_tokens

    chunks_encoded = [
        tokens[i : i + max_chunk_size]
        for i in range(0, total_tokens - max_chunk_size + 1, max_chunk_size - overlap)
    ]
    chunks_encoded.append(tokens[total_tokens - max_chunk_size + 1 :])
    chunks = [encoding.decode(chunk) for chunk in chunks_encoded]
    return chunks, chunks_encoded, total_tokens


class Chunker:
    def __init__(self, encoding: tiktoken.Encoding):
        self._encoding = encoding

    def __call__(
        self, text: str, max_chunk_size: int, overlap: int
    ) -> tuple[list[str], list[list[int]], int]:
        tokens = self._encoding.encode(text)
        total_tokens = len(tokens)
        if total_tokens <= max_chunk_size:
            return [text], [tokens], total_tokens

        chunks = []
        chunks_encoded = []
        for i in range(0, total_tokens - max_chunk_size + 1, max_chunk_size - overlap):
            chunk_encoded = tokens[i : i + max_chunk_size]
            chunks.append(self._encoding.decode(chunk_encoded))
            chunks_encoded.append(chunk_encoded)

        last_chunk_encoded = tokens[total_tokens - max_chunk_size + 1 :]
        chunks_encoded.append(last_chunk_encoded)
        chunks.append(self._encoding.decode(last_chunk_encoded))
        return chunks, chunks_encoded, total_tokens


@pytest.fixture
def data():
    with open("data.txt", "r") as f:
        d = f.read()

    return d


def test_python_chunk_text(benchmark, data):
    encoding = tiktoken.get_encoding("cl100k_base")
    chunks, chunks_encoded, total_tokens = benchmark(
        chunk_text, encoding, data, 1000, 0
    )


def test_python_chunker(benchmark, data):
    encoding = tiktoken.get_encoding("cl100k_base")
    chunker = Chunker(encoding)
    chunks, chunks_encoded, total_tokens = benchmark(chunker, data, 1000, 0)


def test_rust_chunker(benchmark, data):
    chunker = rust_chunker.Chunker()
    chunks, chunks_encoded, total_tokens = benchmark(chunker, data, 1000, 0)


def test_rust_chunker2(benchmark, data):
    chunker = rust_chunker.Chunker2()
    chunks, chunks_encoded, total_tokens = benchmark(chunker, data, 1000, 0)
