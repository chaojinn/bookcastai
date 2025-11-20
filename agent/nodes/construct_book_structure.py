from __future__ import annotations

import hashlib
import json
import logging
import os
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Sequence

from dotenv import load_dotenv
from openai import OpenAI

if TYPE_CHECKING:  # pragma: no cover - typing aid
    from ..epub_agent import EPUBAgentState

logger = logging.getLogger(__name__)

_MODEL_NAME = "tngtech/deepseek-r1t2-chimera:free"

def make_construct_book_structure_node() -> Callable[["EPUBAgentState"], "EPUBAgentState"]:
    def construct_book_structure(state: "EPUBAgentState") -> "EPUBAgentState":
        preview_path_value = state.get("preview_path")
        if not preview_path_value:
            return {}
        chapters = state.get("chapters", [])
        preview_items: List[dict[str, str | int]] = []
        for chapter in chapters:
            if not isinstance(chapter, dict):
                continue
            chapter_number = chapter.get("chapter_number") if isinstance(chapter.get("chapter_number"), int) else None
            title = chapter.get("chapter_title") if isinstance(chapter.get("chapter_title"), str) else ""
            content_text = chapter.get("content_text") if isinstance(chapter.get("content_text"), str) else ""
            trimmed_content = content_text.strip()
            preview_items.append(
                {
                    "chapter_number": chapter_number,
                    "chapter_title": title,
                    "first_200_characters_of_content": trimmed_content[:200],
                    "content_length": len(trimmed_content),
                },
            )
        preview_path = Path(preview_path_value).expanduser()
        try:
            preview_path.parent.mkdir(parents=True, exist_ok=True)
            with preview_path.open("w", encoding="utf-8") as handle:
                json.dump(preview_items, handle, ensure_ascii=False, indent=2)
        except OSError as exc:
            errors = list(state.get("errors", []))
            errors.append(f"Failed to write preview JSON: {exc}")
            return {"errors": errors}

        cache_path = preview_path.parent / "openroute_cache.json"
        selected_numbers = _query_openroute_for_chapters(preview_items, cache_path=cache_path)
        if selected_numbers:
            updated_chapters = [
                chapter
                for chapter in chapters
                if isinstance(chapter, dict)
                and isinstance(chapter.get("chapter_number"), int)
                and chapter["chapter_number"] in selected_numbers
            ]
            if updated_chapters:
                for index, chapter in enumerate(updated_chapters, start=1):
                    chapter["chapter_number"] = index
                return {"chapters": updated_chapters}
            logger.warning("OpenRoute selection returned numbers not matching any chapter.")
        return {}

    return construct_book_structure


def _query_openroute_for_chapters(
    preview_items: Sequence[dict[str, object]],
    *,
    cache_path: Path,
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

    logger.info("OpenRoute chapter selection input: %s", preview_items)

    messages = [
        {
            "role": "system",
            "content": "Output MUST be a JSON array of integers, nothing else.",
        },
        {
            "role": "user",
            "content": json.dumps(
                {
                    "instructions": instructions,
                    "chapters": preview_items,
                },
                ensure_ascii=False,
            ),
        },
    ]

    request_payload = {
        "model": _MODEL_NAME,
        "temperature": 0,
        "messages": messages,
    }
    cached_content = _get_cached_openroute_response(request_payload, cache_path=cache_path)
    if cached_content is not None:
        logger.info("OpenRoute chapter selection cache hit.")
        content = cached_content
    else:
        logger.info("OpenRoute chapter selection request (no cache): %s", request_payload)
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
    logger.info("OpenRoute chapter selection result: %s", cleaned)
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
