"""
Athena Book Library

Reading mode: grounded question answering over local PDF books.

Books far exceed the model's context window, so the whole text cannot be
injected at once. Instead a selected book is extracted once, split into
overlapping word-window chunks, and — per question — only the most relevant
chunks (scored by keyword overlap) are injected, with a strict "answer only
from these excerpts" system prompt.

This path is deliberately separate from the normal Thought pipeline: no tools,
no memory retrieval, no knowledge extraction. Only the book's contents inform
the answer.
"""

import re
from collections import Counter
from pathlib import Path
from typing import Callable, List

from athena.config.settings import get_settings


# Common words that carry no discriminative value for retrieval scoring.
_STOPWORDS = frozenset({
    "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "can",
    "could", "should", "and", "or", "but", "for", "with", "about", "into",
    "of", "to", "in", "on", "at", "by", "as", "it", "this", "that", "these",
    "those", "what", "which", "who", "whom", "how", "when", "where", "why",
    "you", "your", "i", "me", "my", "we", "they", "them", "he", "she", "his",
    "her", "not", "no", "there", "their", "from", "does", "tell",
})


# ── Discovery ─────────────────────────────────────────────────────

def _books_dir() -> Path:
    return Path(get_settings().storage.books_path)


def list_books() -> List[Path]:
    """Return the PDF files available in the books directory, sorted by name."""
    directory = _books_dir()
    if not directory.exists():
        return []
    return sorted(directory.glob("*.pdf"))


# ── Extraction & chunking ─────────────────────────────────────────

def extract_text(pdf_path) -> str:
    """Extract all text from a PDF. Returns '' if nothing can be read.

    Scanned/image-only PDFs yield little or no text; the caller should treat an
    empty result as "unreadable" rather than a valid empty book.
    """
    from pypdf import PdfReader

    reader = PdfReader(str(pdf_path))
    parts: List[str] = []
    for page in reader.pages:
        try:
            parts.append(page.extract_text() or "")
        except Exception:
            continue
    return "\n".join(parts)


def chunk_text(text: str, target_words: int = 180, overlap_words: int = 40) -> List[str]:
    """Split text into overlapping fixed-size word windows.

    Word windows are robust to the messy, inconsistent whitespace produced by
    PDF extraction (paragraph breaks are frequently lost). Overlap keeps
    sentences that straddle a boundary retrievable from at least one chunk.
    """
    words = text.split()
    if not words:
        return []
    if target_words <= 0:
        target_words = 180
    step = max(1, target_words - max(0, overlap_words))

    chunks: List[str] = []
    for start in range(0, len(words), step):
        window = words[start:start + target_words]
        if window:
            chunks.append(" ".join(window))
        if start + target_words >= len(words):
            break
    return chunks


# ── Retrieval ─────────────────────────────────────────────────────

def _terms(text: str) -> List[str]:
    return [t for t in re.findall(r"[a-z0-9]+", text.lower()) if len(t) > 2 and t not in _STOPWORDS]


def retrieve_relevant(
    chunks: List[str],
    question: str,
    count_tokens: Callable[[str], int],
    budget_tokens: int,
    max_chunks: int = 6,
) -> List[str]:
    """Return the chunks most relevant to the question that fit the budget.

    Chunks are scored by summed term-frequency of the question's discriminative
    words. The highest-scoring chunks are packed until the token budget or
    max_chunks is reached, then returned in original document order for
    readability. If no chunk matches (or the question has no discriminative
    terms), the opening chunks are returned so questions like "what is this
    book about" still get grounded context.
    """
    if not chunks:
        return []

    question_terms = _terms(question)

    scored = []
    for index, chunk in enumerate(chunks):
        if question_terms:
            counts = Counter(re.findall(r"[a-z0-9]+", chunk.lower()))
            score = sum(counts.get(term, 0) for term in question_terms)
        else:
            score = 0
        scored.append((score, index, chunk))

    has_match = any(score > 0 for score, _, _ in scored)

    if has_match:
        # Highest score first; original order breaks ties.
        scored.sort(key=lambda item: (-item[0], item[1]))
        candidates = [(idx, chunk) for score, idx, chunk in scored if score > 0]
    else:
        # No lexical match — fall back to the opening of the book.
        candidates = [(idx, chunk) for _, idx, chunk in scored]

    selected = []
    used = 0
    for index, chunk in candidates:
        cost = count_tokens(chunk)
        if selected and used + cost > budget_tokens:
            continue
        selected.append((index, chunk))
        used += cost
        if len(selected) >= max_chunks or used >= budget_tokens:
            break

    selected.sort(key=lambda item: item[0])
    return [chunk for _, chunk in selected]


# ── Prompt assembly & answering ───────────────────────────────────

def build_book_prompt(passages: List[str], question: str) -> str:
    """Assemble the reading-mode user prompt from passages and the question."""
    excerpts = "\n\n- - -\n\n".join(passages)
    return (
        "Book excerpts:\n\n"
        f"{excerpts}\n\n"
        "====================\n\n"
        f"Question: {question}"
    )


def answer_from_book(provider, chunks: List[str], question: str) -> str:
    """Answer a question grounded strictly in the given book chunks.

    Uses the provider directly (no Thought pipeline, no tools, no memory). The
    relevant passages are retrieved to fit the provider's context window, then
    sent with the strict book-mode system prompt.
    """
    from athena.prompt.loader import PromptLoader

    context_window = provider.get_context_window()
    generation_reserve = int(context_window * 0.25)

    system_prompt = PromptLoader.get_system_prompt("book")
    overhead = (
        provider.count_tokens(system_prompt)
        + provider.count_tokens(question)
        + 128  # headers / formatting slack
    )
    budget = max(256, context_window - generation_reserve - overhead)

    passages = retrieve_relevant(chunks, question, provider.count_tokens, budget)
    if not passages:
        return "I couldn't find anything about that in this book."

    prompt = build_book_prompt(passages, question)
    return provider.generate(prompt, system=system_prompt)
