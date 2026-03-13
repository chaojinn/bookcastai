from __future__ import annotations

import hashlib
import json
import logging
import os
import re
from html.parser import HTMLParser
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Sequence

from dotenv import load_dotenv
from openai import OpenAI

if TYPE_CHECKING:  # pragma: no cover - typing aid
    from ..epub_agent import EPUBAgentState

logger = logging.getLogger(__name__)

_MODEL_NAME = "openai/gpt-5.2"
#_MODEL_NAME = "qwen/qwen3-coder"

_BLOCK_LEVEL_TAGS = {
    "article", "aside", "blockquote", "br", "div", "footer",
    "h1", "h2", "h3", "h4", "h5", "h6", "header", "hr",
    "li", "main", "nav", "p", "pre", "section", "ul", "ol",
}


class _ContentStripper(HTMLParser):
    """Strip HTML tags from content."""

    def __init__(self) -> None:
        super().__init__()
        self._parts: List[str] = []

    def clean(self, html_text: str) -> str:
        if not isinstance(html_text, str):
            return ""
        self.reset()
        self._parts.clear()
        self.feed(html_text)
        self.close()
        text = "".join(self._parts).strip()
        text = text.replace("\n", " ")
        cleaned = " ".join(text.split())
        cleaned = re.sub(r"\([^)]*\)", "", cleaned)
        cleaned = " ".join(cleaned.split())
        return cleaned

    def handle_starttag(self, tag: str, attrs: object) -> None:
        if tag.lower() in _BLOCK_LEVEL_TAGS:
            self._append_separator()

    def handle_startendtag(self, tag: str, attrs: object) -> None:
        if tag.lower() in _BLOCK_LEVEL_TAGS:
            self._append_separator()

    def handle_endtag(self, tag: str) -> None:
        if tag.lower() in _BLOCK_LEVEL_TAGS:
            self._append_separator()

    def handle_data(self, data: str) -> None:
        if data:
            self._parts.append(data)

    def _append_separator(self) -> None:
        if not self._parts:
            return
        last = self._parts[-1]
        if not last:
            return
        last_non_space = None
        for ch in reversed(last):
            if not ch.isspace():
                last_non_space = ch
                break
        if last_non_space is None:
            return
        if last_non_space not in {".", "?", "!", ":", ";", ","}:
            self._parts[-1] = last.rstrip() + "."
        self._parts.append(" ")


def make_construct_book_structure_node(
    cache_path: Optional[Path] = None,
) -> Callable[["EPUBAgentState"], "EPUBAgentState"]:
    stripper = _ContentStripper()

    def construct_book_structure_new(state: "EPUBAgentState") -> "EPUBAgentState":
        raw_chapters = state.get("chapters", [])
        errors: List[str] = list(state.get("errors", []))

        # Strip HTML from each chapter
        stripped_chapters = []
        for chapter in raw_chapters:
            if not isinstance(chapter, dict):
                continue
            stripped = dict(chapter)
            stripped["content_text"] = stripper.clean(chapter.get("content_text", ""))
            stripped_chapters.append(stripped)

        # Merge short chapters (< 100 chars) into the next chapter
        stripped_chapters = _merge_short_chapters(stripped_chapters)

        # Build preview items for LLM
        preview_items: List[dict[str, Any]] = []
        for chapter in stripped_chapters:
            chapter_number = chapter.get("chapter_number")
            title = chapter.get("chapter_title", "") if isinstance(chapter.get("chapter_title"), str) else ""
            content_text = chapter.get("content_text", "") if isinstance(chapter.get("content_text"), str) else ""
            trimmed = content_text.strip()
            preview_items.append(
                {
                    "chapter_number": chapter_number,
                    "chapter_title": title,
                    "first_200_characters_of_content": trimmed[:200],
                    "content_length": len(trimmed),
                }
            )

        selected_numbers = _query_openroute_for_chapters(preview_items, cache_path=cache_path)
        if selected_numbers:
            filtered = [
                chapter
                for chapter in stripped_chapters
                if isinstance(chapter.get("chapter_number"), int)
                and chapter["chapter_number"] in selected_numbers
            ]
            if filtered:
                for index, chapter in enumerate(filtered, start=1):
                    chapter["chapter_number"] = index
                for chapter in filtered:
                    title = chapter.get("chapter_title", "")
                    content = chapter.get("content_text", "")
                    logger.info(
                        "Chapter %s: '%s' (length=%s)",
                        chapter.get("chapter_number"),
                        title,
                        len(content),
                    )
                return {"chapters": filtered, "errors": errors}
            logger.warning("OpenRoute selection returned numbers not matching any chapter.")

        return {"chapters": stripped_chapters, "errors": errors}

    return construct_book_structure_new


def _merge_short_chapters(chapters: List[dict[str, Any]]) -> List[dict[str, Any]]:
    """Merge any chapter with < 100 chars of content into the next chapter,
    using the next chapter's title. Renumbers sequentially after merging."""
    merged: List[dict[str, Any]] = []
    carry_content = ""
    carry_href = ""

    for chapter in chapters:
        content = chapter.get("content_text", "") if isinstance(chapter.get("content_text"), str) else ""
        if len(content.strip()) < 100:
            carry_content = (carry_content + " " + content).strip() if carry_content else content
            if not carry_href:
                carry_href = chapter.get("href", "")
            logger.debug(
                "Merging short chapter %s ('%s', len=%s) into next chapter.",
                chapter.get("chapter_number"),
                chapter.get("chapter_title", ""),
                len(content.strip()),
            )
            continue

        # This chapter absorbs any carried content; it keeps its own title
        combined = (carry_content + " " + content).strip() if carry_content else content
        new_chapter = dict(chapter)
        new_chapter["content_text"] = combined
        if carry_href and not new_chapter.get("href"):
            new_chapter["href"] = carry_href
        merged.append(new_chapter)
        carry_content = ""
        carry_href = ""

    # If trailing carry remains, append to last chapter
    if carry_content and merged:
        last = merged[-1]
        last_content = last.get("content_text", "") if isinstance(last.get("content_text"), str) else ""
        last["content_text"] = (last_content + " " + carry_content).strip()

    # Renumber sequentially
    for idx, chapter in enumerate(merged, start=1):
        chapter["chapter_number"] = idx

    return merged


def _query_openroute_for_chapters(
    preview_items: Sequence[dict[str, Any]],
    *,
    cache_path: Optional[Path],
) -> List[int]:
    if not preview_items:
        return []

    instructions = (
        "there is an array of chapter data from a book, it contains chapter number, title, first 200 characters of "
        "content, and content length, choose which chapters should be included in the audio book.\n"
        "remove all introduction, preface, preamble, foreword copyright etc before first chapter of the content\n"
        "remove all epilogue, appendix and similar at the end of the book after last chapter of the content\n"
        "return an json array of chapter numbers which should be included in the audio book\n"
        "return must be a valid json array with nothing else"
    )

    logger.debug("OpenRoute chapter selection input: %s", preview_items)

    messages = [
        {
            "role": "system",
            "content": "Output MUST be a JSON array of integers, nothing else.",
        },
        {
            "role": "user",
            "content": json.dumps(
                {"instructions": instructions, "chapters": preview_items},
                ensure_ascii=False,
            ),
        },
    ]

    request_payload = {
        "model": _MODEL_NAME,
        "temperature": 0,
        "messages": messages,
    }

    if cache_path is not None:
        cached_content = _get_cached_openroute_response(request_payload, cache_path=cache_path)
        if cached_content is not None:
            logger.info("OpenRoute chapter selection cache hit.")
            content = cached_content
        else:
            content = None
    else:
        content = None

    if content is None:
        logger.debug("OpenRoute chapter selection request (no cache): %s", request_payload)
        load_dotenv()
        api_key = os.getenv("OPENROUTE_API_KEY")
        if not api_key:
            logger.warning("OPENROUTE_API_KEY is not set; unable to select chapters for audio book.")
            return []

        client = OpenAI(api_key=api_key, base_url="https://openrouter.ai/api/v1")
        try:
            response = client.chat.completions.create(**request_payload)
        except Exception as exc:  # pragma: no cover - depends on network
            logger.error("OpenRoute chapter selection failed: %s", exc)
            return []

        content = response.choices[0].message.content if response.choices else None
        if not content:
            logger.error("OpenRoute chapter selection returned empty content.")
            return []

        if cache_path is not None:
            _store_openroute_response(request_payload, content, cache_path=cache_path)

    try:
        raw_numbers = json.loads(content)
    except json.JSONDecodeError:
        logger.error("OpenRoute selection did not return valid JSON. Content: %s", content)
        return []

    logger.info("OpenRoute chapter selection output: %s", raw_numbers)

    cleaned: List[int] = []
    seen: set[int] = set()
    if isinstance(raw_numbers, list):
        for value in raw_numbers:
            number: int | None = None
            if isinstance(value, int):
                number = value
            elif isinstance(value, str):
                stripped = value.strip()
                if stripped.isdigit():
                    number = int(stripped)
            if number is None or number <= 0 or number in seen:
                continue
            seen.add(number)
            cleaned.append(number)

    if not cleaned:
        logger.warning("OpenRoute selection response did not contain usable chapter numbers.")
    logger.debug("OpenRoute chapter selection result: %s", cleaned)
    return cleaned


def _get_cached_openroute_response(payload: Dict[str, Any], *, cache_path: Path) -> str | None:
    cache = _load_openroute_cache(cache_path=cache_path)
    key = _openroute_cache_key(payload)
    entry = cache.get(key)
    if not isinstance(entry, dict):
        return None
    response = entry.get("response")
    return response if isinstance(response, str) else None


def _store_openroute_response(payload: Dict[str, Any], response: str, *, cache_path: Path) -> None:
    cache = _load_openroute_cache(cache_path=cache_path)
    key = _openroute_cache_key(payload)
    cache[key] = {"request": payload, "response": response}
    _save_openroute_cache(cache, cache_path=cache_path)


def _load_openroute_cache(*, cache_path: Path) -> Dict[str, Any]:
    if not cache_path.exists():
        return {}
    try:
        with cache_path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
            return data if isinstance(data, dict) else {}
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("Unable to read OpenRoute cache: %s", exc)
        return {}


def _save_openroute_cache(cache: Dict[str, Any], *, cache_path: Path) -> None:
    try:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with cache_path.open("w", encoding="utf-8") as handle:
            json.dump(cache, handle, ensure_ascii=False, indent=2)
    except OSError as exc:
        logger.warning("Unable to write OpenRoute cache: %s", exc)


def _openroute_cache_key(payload: Dict[str, Any]) -> str:
    serialized = json.dumps(payload, ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()
