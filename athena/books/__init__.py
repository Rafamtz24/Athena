"""Book reading mode — grounded question answering over local PDFs."""

from athena.books.library import (
    answer_from_book,
    build_book_prompt,
    chunk_text,
    extract_text,
    list_books,
    retrieve_relevant,
)

__all__ = [
    "answer_from_book",
    "build_book_prompt",
    "chunk_text",
    "extract_text",
    "list_books",
    "retrieve_relevant",
]
