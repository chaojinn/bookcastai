from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Dict, List

import nltk
from nltk.data import load as nltk_data_load
from nltk.tokenize import PunktSentenceTokenizer
from .text_processing import EPUBTextProcessor

if TYPE_CHECKING:  # pragma: no cover - typing aid
    from ..epub_agent import ChapterPayload, EPUBAgentState


def _load_sentence_tokenizer() -> PunktSentenceTokenizer:
    try:
        return nltk_data_load("tokenizers/punkt/english.pickle")
    except LookupError:
        # Attempt to download missing punkt resources automatically.
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


_SENTENCE_TOKENIZER: PunktSentenceTokenizer = _load_sentence_tokenizer()


def _split_text_into_chunks(text: str, chunk_size: int) -> List[str]:
    size = max(1, chunk_size)
    return [text[index : index + size] for index in range(0, len(text), size)]


def _segment_text_into_sentences(text: str) -> List[str]:
    stripped = text.strip()
    if not stripped:
        return []
    return _SENTENCE_TOKENIZER.tokenize(stripped)


def _split_text_by_sentences(text: str, chunk_size: int) -> List[str]:
    size = max(1, chunk_size)
    segments = _segment_text_into_sentences(text)
    if not segments:
        return _split_text_into_chunks(text, size)

    chunks: List[str] = []
    current_parts: List[str] = []
    current_len = 0

    def flush_current() -> None:
        nonlocal current_parts, current_len
        chunk = " ".join(part.strip() for part in current_parts if part.strip()).strip()
        if chunk:
            chunks.append(chunk)
        current_parts = []
        current_len = 0

    for segment in segments:
        stripped_segment = segment.strip()
        if not stripped_segment:
            continue

        segment_len = len(stripped_segment)
        if segment_len > size:
            flush_current()
            for piece in _split_text_into_chunks(stripped_segment, size):
                cleaned = piece.strip()
                if cleaned:
                    chunks.append(cleaned)
            continue

        projected_len = current_len + segment_len + (1 if current_parts else 0)
        if projected_len <= size:
            current_parts.append(stripped_segment)
            current_len = projected_len
            continue

        flush_current()
        current_parts.append(stripped_segment)
        current_len = segment_len

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

        epub_path = state.get("epub_path")
        book_title = ""
        if isinstance(epub_path, str):
            try:
                book_title = Path(epub_path).expanduser().resolve().parent.name
            except OSError:
                book_title = Path(epub_path).parent.name
        ai_enabled = bool(getattr(text_processor, "_ai_extract_text", False))
        ai_changes_log: List[Dict[str, Any]] = []
        chapter_change_index: Dict[int, Dict[str, Any]] = {}

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
            chapter_chunks = _split_text_by_sentences(content, effective_chunk_size)
            chunks, chunk_change_entries = text_processor.convert_chunks_for_tts(chapter, chapter_chunks)
            existing_chunks = chapter.get("chunks")
            if isinstance(existing_chunks, list) and existing_chunks == chunks:
                updated_chapters.append(chapter)
                continue
            chapter_copy: ChapterPayload = dict(chapter)
            chapter_copy["chunks"] = chunks
            updated_chapters.append(chapter_copy)
            modified = True

            chapter_number = chapter.get("chapter_number")
            if chunk_change_entries and isinstance(chapter_number, int):
                chapter_entry = chapter_change_index.get(chapter_number)
                if chapter_entry is None:
                    chapter_entry = {"chapter_number": chapter_number, "chunks": []}
                    chapter_change_index[chapter_number] = chapter_entry
                    ai_changes_log.append(chapter_entry)
                chapter_entry["chunks"].extend(chunk_change_entries)

        if ai_enabled and book_title:
            changes_path = Path("data") / book_title / "ai_changes.json"
            changes_path.parent.mkdir(parents=True, exist_ok=True)
            with changes_path.open("w", encoding="utf-8") as handle:
                json.dump(ai_changes_log, handle, indent=2, ensure_ascii=False)

        return {"chapters": updated_chapters} if modified else {}

    return chunk_chapter_content
