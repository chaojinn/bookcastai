from __future__ import annotations

from typing import TYPE_CHECKING, Callable, List

from .text_processing import EPUBTextProcessor

if TYPE_CHECKING:  # pragma: no cover - typing aid
    from ..epub_agent import ChapterPayload, EPUBAgentState


_SENTENCE_TERMINATORS = ".?!"
_COMMON_ABBREVIATIONS = {
    "dr",
    "mr",
    "mrs",
    "ms",
    "prof",
    "sir",
    "jr",
    "rev",
}


def _split_text_into_chunks(text: str, chunk_size: int) -> List[str]:
    size = max(1, chunk_size)
    return [text[index : index + size] for index in range(0, len(text), size)]


def _looks_like_abbreviation(text: str, punct_index: int) -> bool:
    if punct_index <= 0 or punct_index > len(text):
        return False
    word_end = punct_index
    word_start = word_end
    while word_start > 0 and text[word_start - 1].isalpha():
        word_start -= 1
    if word_start == word_end:
        return False
    candidate = text[word_start:word_end].lower()
    return candidate in _COMMON_ABBREVIATIONS


def _segment_text_by_punctuation(text: str) -> List[str]:
    segments: List[str] = []
    start = 0
    idx = 0
    length = len(text)
    while idx < length:
        char = text[idx]
        if char in _SENTENCE_TERMINATORS:
            if char == "." and _looks_like_abbreviation(text, idx):
                idx += 1
                continue
            idx += 1
            while idx < length and text[idx].isspace():
                idx += 1
            segments.append(text[start:idx])
            start = idx
            continue
        idx += 1
    if start < length:
        segments.append(text[start:])
    return [segment for segment in segments if segment.strip()]


def _split_text_by_punctuation(text: str, chunk_size: int) -> List[str]:
    size = max(1, chunk_size)
    segments = _segment_text_by_punctuation(text)
    if not segments:
        return _split_text_into_chunks(text, size)

    chunks: List[str] = []
    current_parts: List[str] = []
    current_len = 0

    def flush_current() -> None:
        nonlocal current_parts, current_len
        chunk = "".join(current_parts).strip()
        if chunk:
            chunks.append(chunk)
        current_parts = []
        current_len = 0

    for segment in segments:
        if len(segment) > size:
            flush_current()
            for piece in _split_text_into_chunks(segment, size):
                cleaned = piece.strip()
                if cleaned:
                    chunks.append(cleaned)
            continue

        if current_len + len(segment) <= size:
            current_parts.append(segment)
            current_len += len(segment)
            continue

        flush_current()
        current_parts.append(segment)
        current_len = len(segment)

    flush_current()
    return chunks


def make_chunk_chapter_content_node(
    *,
    text_processor: EPUBTextProcessor,
    default_chunk_size: int,
) -> Callable[["EPUBAgentState"], "EPUBAgentState"]:
    def chunk_chapter_content(state: "EPUBAgentState") -> "EPUBAgentState":
        chapters = state.get("chapters", [])
        if not chapters:
            return {}

        state_chunk_size = state.get("chunk_size")
        effective_chunk_size = (
            max(1, int(state_chunk_size)) if isinstance(state_chunk_size, int) else default_chunk_size
        )

        updated_chapters: List[ChapterPayload] = []
        modified = False
        for chapter in chapters:
            if not isinstance(chapter, dict):
                updated_chapters.append(chapter)
                continue
            content = chapter.get("content_text")
            if not isinstance(content, str) or not content:
                updated_chapters.append(chapter)
                continue
            chunks = _split_text_by_punctuation(content, effective_chunk_size)
            chunks = text_processor.convert_chunks_for_tts(chapter, chunks)
            existing_chunks = chapter.get("chunks")
            if isinstance(existing_chunks, list) and existing_chunks == chunks:
                updated_chapters.append(chapter)
                continue
            chapter_copy: ChapterPayload = dict(chapter)
            chapter_copy["chunks"] = chunks
            updated_chapters.append(chapter_copy)
            modified = True
        return {"chapters": updated_chapters} if modified else {}

    return chunk_chapter_content
