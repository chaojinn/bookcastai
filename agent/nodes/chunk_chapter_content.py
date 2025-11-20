from __future__ import annotations

import json
import logging
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Dict, List

import nltk
from nltk.data import load as nltk_data_load
from nltk.tokenize import PunktSentenceTokenizer
from .text_processing import EPUBTextProcessor

if TYPE_CHECKING:  # pragma: no cover - typing aid
    from ..epub_agent import ChapterPayload, EPUBAgentState

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


def _split_text_by_sentences(
    text: str,
    chunk_size: int,
    *,
    chapter_number: int | None = None,
) -> List[str]:
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
                logger.warning(
                    "Chunk length %s exceeds max %s%s",
                    len(chunk),
                    upper_size,
                    f" (chapter {chapter_number})" if chapter_number is not None else "",
                )
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
            chunks.append(stripped_sentence)
            logger.warning(
                "Chunk length %s exceeds max %s%s",
                sentence_len,
                upper_size,
                f" (chapter {chapter_number})" if chapter_number is not None else "",
            )
            continue

        projected_len = current_len + sentence_len + (1 if current_parts else 0)
        if projected_len > upper_size and current_len < size:
            # Prefer keeping sentences together over emitting very small chunks; warn on overflow.
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


def _build_sentences_payload(
    *,
    book_title: str,
    epub_path: str | Path | None,
    chapters: List["ChapterPayload"],
) -> Dict[str, Any]:
    try:
        book_path = Path(epub_path).expanduser().resolve() if epub_path else None
    except OSError:
        book_path = None

    chapter_entries: List[Dict[str, Any]] = []
    for index, chapter in enumerate(chapters):
        if not isinstance(chapter, dict):
            continue
        sentences = split_sentences(chapter.get("content_text") or "")
        chapter_entries.append(
            {
                "chapter_index": index,
                "chapter_title": chapter.get("chapter_title"),
                "chapter_number": chapter.get("chapter_number"),
                "sentences": sentences,
            }
        )

    return {
        "book_title": book_title,
        "book_path": str(book_path) if book_path else "",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "chapters": chapter_entries,
    }


def _write_sentences_json(book_title: str, payload: Dict[str, Any]) -> None:
    target_path = Path("data") / book_title / "sentences.json"
    target_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with target_path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)
    except OSError as exc:
        logger.warning("Failed to write sentences.json: %s", exc)


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
        sentences_payload: Dict[str, Any] | None = None

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
            chapter_chunks = _split_text_by_sentences(
                content,
                effective_chunk_size,
                chapter_number=chapter.get("chapter_number") if isinstance(chapter.get("chapter_number"), int) else None,
            )
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

        if book_title:
            sentences_payload = _build_sentences_payload(
                book_title=book_title,
                epub_path=epub_path,
                chapters=updated_chapters if modified else chapters,
            )

        if ai_enabled and book_title:
            changes_path = Path("data") / book_title / "ai_changes.json"
            changes_path.parent.mkdir(parents=True, exist_ok=True)
            with changes_path.open("w", encoding="utf-8") as handle:
                json.dump(ai_changes_log, handle, indent=2, ensure_ascii=False)

        if sentences_payload:
            _write_sentences_json(book_title, sentences_payload)

        return {"chapters": updated_chapters} if modified else {}

    return chunk_chapter_content
