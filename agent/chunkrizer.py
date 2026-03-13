from __future__ import annotations

import logging
import re
from typing import List

import nltk
from nltk.data import load as nltk_data_load
from nltk.tokenize import PunktSentenceTokenizer

logger = logging.getLogger(__name__)

_ABBREVIATIONS = {
    "Mr.",
    "Mrs.",
    "Ms.",
    "Dr.",
    "Prof.",
    "Sr.",
    "Jr.",
    "Hon.",
    "Rev.",
    "Pres.",
    "Gov.",
    "Sen.",
    "Rep.",
    "St.",
    "a.m.",
    "p.m.",
    "BC.",
    "B.C.E.",
    "A.D.",
    "approx.",
    "est.",
    "ft.",
    "in.",
    "oz.",
    "lb.",
    "gal.",
    "qt.",
    "pt.",
    "sq.",
    "cu.",
    "mph.",
    "km.",
    "cm.",
    "mm.",
    "mg.",
    "ml.",
    "etc.",
    "e.g.",
    "i.e.",
    "vs.",
    "et al.",
    "fig.",
    "cf.",
    "vol.",
    "vols.",
    "no.",
    "nos.",
    "ch.",
    "chap.",
    "ed.",
    "eds.",
    "trans.",
    "ref.",
    "refs.",
    "al.",
    "Inc.",
    "Ltd.",
    "Co.",
    "Corp.",
    "LLC.",
    "Bros.",
    "Dept.",
    "Div.",
    "Mfg.",
    "Est.",
    "Mt.",
    "Blvd.",
    "Rd.",
    "Ave.",
    "Ln.",
    "Hwy.",
    "Sq.",
    "Apt.",
    "Fl.",
    "Gen.",
    "Col.",
    "Lt.",
    "Cmdr.",
    "Capt.",
    "Sgt.",
    "Pvt.",
    "Maj.",
    "misc.",
    "min.",
    "sec.",
    "dept.",
    "calc.",
    "max.",
    "seq.",
    "alt.",
    "equiv.",
    "var.",
    "orig.",
    "anon.",
}


def _load_sentence_tokenizer() -> PunktSentenceTokenizer:
    try:
        return nltk_data_load("tokenizers/punkt/english.pickle")
    except LookupError:
        for resource in ("punkt", "punkt_tab"):
            try:
                nltk.download(resource, quiet=True)
            except Exception:
                continue
        try:
            return nltk_data_load("tokenizers/punkt/english.pickle")
        except LookupError as exc:  # pragma: no cover - depends on env
            raise RuntimeError(
                "NLTK punkt tokenizer unavailable. Run python -m nltk.downloader punkt punkt_tab."
            ) from exc


def _configure_abbreviations(tokenizer: PunktSentenceTokenizer) -> PunktSentenceTokenizer:
    tokenizer._params.abbrev_types.update(
        {abbr.rstrip(".").lower() for abbr in _ABBREVIATIONS}
    )
    return tokenizer


_SENTENCE_TOKENIZER: PunktSentenceTokenizer = _configure_abbreviations(_load_sentence_tokenizer())


def _strip_parentheticals(text: str) -> str:
    cleaned = text or ""
    while True:
        updated = re.sub(r"\([^()]*\)", "", cleaned)
        if updated == cleaned:
            break
        cleaned = updated
    return cleaned


def _collapse_continuous_dots(text: str) -> str:
    return re.sub(r"\.\s*(?:\.\s*)+", ".", text or "")


def split_sentences(text: str, *, strip_parentheticals: bool = True) -> List[str]:
    cleaned_text = _strip_parentheticals(text) if strip_parentheticals else (text or "")
    normalized_text = _collapse_continuous_dots(cleaned_text)
    stripped = normalized_text.strip()
    if not stripped:
        return []
    sentences = _SENTENCE_TOKENIZER.tokenize(stripped)
    return [sentence.strip() for sentence in sentences if sentence.strip()]


def _split_text_into_chunks(text: str, chunk_size: int) -> List[str]:
    size = max(1, chunk_size)
    return [text[index : index + size] for index in range(0, len(text), size)]


def _split_chunk_at_delimiter(text: str, upper_size: int) -> List[str]:
    """Split an oversized chunk at the comma/semicolon nearest the midpoint."""
    stripped = text.strip()
    if len(stripped) <= upper_size:
        return [stripped] if stripped else []

    mid = len(stripped) // 2
    split_points = [idx for idx, ch in enumerate(stripped) if ch in {",", ";"}]
    if not split_points:
        return [stripped] if stripped else []

    split_idx = min(split_points, key=lambda idx: abs(idx - mid))
    left = stripped[: split_idx + 1].strip()
    right = stripped[split_idx + 1 :].strip()

    parts: List[str] = []
    if left:
        parts.append(left)
    if right:
        parts.append(right)
    return parts


def chunk_text(text: str, chunk_size: int) -> List[str]:
    """Split text into chunks of approximately chunk_size characters, respecting sentence boundaries.

    Parameters
    ----------
    text:
        The input text to chunk.
    chunk_size:
        Target maximum character length per chunk.

    Returns
    -------
    List[str]
        Array of text chunks.
    """
    size = max(1, chunk_size)
    upper_size = size * 2
    sentences = split_sentences(text)
    if not sentences:
        return _split_text_into_chunks(text, size)

    chunks: List[str] = []
    current_parts: List[str] = []
    current_len = 0

    def flush_current() -> None:
        nonlocal current_parts, current_len
        chunk = " ".join(part.strip() for part in current_parts if part.strip()).strip()
        if chunk:
            if len(chunk) > upper_size:
                split_chunks = _split_chunk_at_delimiter(chunk, upper_size) or [chunk]
                for piece in split_chunks:
                    if len(piece) > upper_size:
                        logger.warning("Chunk length %s exceeds max %s", len(piece), upper_size)
                    chunks.append(piece)
            else:
                chunks.append(chunk)
        current_parts = []
        current_len = 0

    for sentence in sentences:
        stripped_sentence = sentence.strip()
        if not stripped_sentence:
            continue
        sentence_len = len(stripped_sentence)

        if sentence_len > upper_size:
            flush_current()
            split_chunks = _split_chunk_at_delimiter(stripped_sentence, upper_size) or [stripped_sentence]
            for piece in split_chunks:
                if len(piece) > upper_size:
                    logger.warning("Chunk length %s exceeds max %s", len(piece), upper_size)
                chunks.append(piece)
            continue

        projected_len = current_len + sentence_len + (1 if current_parts else 0)
        if projected_len > upper_size and current_len < size:
            current_parts.append(stripped_sentence)
            current_len = projected_len
            flush_current()
            continue

        if projected_len > upper_size:
            flush_current()
            current_parts.append(stripped_sentence)
            current_len = sentence_len
            continue

        current_parts.append(stripped_sentence)
        current_len = projected_len

    flush_current()
    return chunks
