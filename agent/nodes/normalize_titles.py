from __future__ import annotations

import hashlib
import json
import logging
import os
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional

from dotenv import load_dotenv
from openai import OpenAI

if TYPE_CHECKING:  # pragma: no cover - typing aid
    from ..epub_agent import EPUBAgentState

logger = logging.getLogger(__name__)

_MODEL_NAME = "openai/gpt-5.2"


def make_normalize_titles_node(
    cache_path: Optional[Path] = None,
) -> Callable[["EPUBAgentState"], "EPUBAgentState"]:
    def normalize_titles_new(state: "EPUBAgentState") -> "EPUBAgentState":
        chapters: List[Dict[str, Any]] = list(state.get("chapters") or [])
        errors: List[str] = list(state.get("errors") or [])

        if not chapters:
            return {"chapters": chapters, "errors": errors}

        title_inputs = [
            {
                "chapter_number": chapter.get("chapter_number", idx + 1),
                "chapter_title": chapter.get("chapter_title", "")
                if isinstance(chapter.get("chapter_title"), str)
                else "",
            }
            for idx, chapter in enumerate(chapters)
        ]

        normalized_titles = _normalize_titles_with_llm(title_inputs, errors, cache_path=cache_path)
        if normalized_titles is None:
            return {"chapters": chapters, "errors": errors}

        updated_chapters = []
        for chapter, norm_title in zip(chapters, normalized_titles):
            chapter_copy = dict(chapter)
            chapter_copy["chapter_title"] = norm_title
            updated_chapters.append(chapter_copy)

        return {"chapters": updated_chapters, "errors": errors}

    return normalize_titles_new


def _normalize_titles_with_llm(
    title_inputs: List[Dict[str, Any]],
    errors: List[str],
    *,
    cache_path: Optional[Path],
) -> Optional[List[str]]:
    instructions = (
        "Normalize each chapter title to the format 'Chapter X: Title' where X is the chapter number.\n"
        "Rules:\n"
        "- Convert Roman numerals in titles to digits (e.g. 'Chapter III' -> 'Chapter 3: ...').\n"
        "- If the original title is empty, return just 'Chapter X' with no colon or trailing text.\n"
        "- Preserve the rest of the title wording after normalizing the number.\n"
        "- Do not convert special chapter names: prolog, prologue, preface, introduction, foreword, "
        "preamble, prelude, epilog, epilogue, conclusion, coda, afterword, postscript, finale — return these as-is.\n"
        "Return a JSON array of strings in the same order as provided, nothing else."
    )

    messages = [
        {
            "role": "system",
            "content": "You normalize chapter titles. Output MUST be a JSON array of strings, nothing else.",
        },
        {
            "role": "user",
            "content": json.dumps(
                {"instructions": instructions, "chapters": title_inputs},
                ensure_ascii=False,
            ),
        },
    ]

    request_payload: Dict[str, Any] = {
        "model": _MODEL_NAME,
        "temperature": 0,
        "messages": messages,
    }

    content: Optional[str] = None
    if cache_path is not None:
        cached = _get_cached_response(request_payload, cache_path=cache_path)
        if cached is not None:
            logger.info("Title normalization cache hit.")
            content = cached

    if content is None:
        logger.info("Title normalization LLM request (no cache): %s", title_inputs)
        load_dotenv()
        api_key = os.getenv("OPENROUTE_API_KEY")
        if not api_key:
            msg = "OPENROUTE_API_KEY is not set; unable to normalize chapter titles."
            logger.warning(msg)
            errors.append(msg)
            return None

        client = OpenAI(api_key=api_key, base_url="https://openrouter.ai/api/v1")
        try:
            response = client.chat.completions.create(**request_payload)
        except Exception as exc:  # pragma: no cover - depends on network
            logger.error("Title normalization LLM call failed: %s", exc)
            errors.append(f"Title normalization LLM call failed: {exc}")
            return None

        content = response.choices[0].message.content if response.choices else None
        if not content:
            logger.error("Title normalization returned empty content.")
            errors.append("Title normalization returned empty content.")
            return None

        if cache_path is not None:
            _store_cached_response(request_payload, content, cache_path=cache_path)

    try:
        normalized = json.loads(content)
    except json.JSONDecodeError:
        logger.error("Title normalization returned invalid JSON: %s", content)
        errors.append("Title normalization returned invalid JSON.")
        return None

    if not isinstance(normalized, list) or len(normalized) != len(title_inputs):
        logger.error(
            "Title normalization returned unexpected result (expected %d items): %s",
            len(title_inputs),
            normalized,
        )
        errors.append("Title normalization returned unexpected result.")
        return None

    result: List[str] = []
    for item in normalized:
        if not isinstance(item, str):
            logger.error("Title normalization produced a non-string item: %s", item)
            errors.append("Title normalization produced a non-string item.")
            return None
        result.append(item)

    logger.info("Title normalization result: %s", result)
    return result


def _get_cached_response(payload: Dict[str, Any], *, cache_path: Path) -> Optional[str]:
    cache = _load_cache(cache_path=cache_path)
    key = _cache_key(payload)
    entry = cache.get(key)
    if not isinstance(entry, dict):
        return None
    response = entry.get("response")
    return response if isinstance(response, str) else None


def _store_cached_response(payload: Dict[str, Any], response: str, *, cache_path: Path) -> None:
    cache = _load_cache(cache_path=cache_path)
    key = _cache_key(payload)
    cache[key] = {"request": payload, "response": response}
    _save_cache(cache, cache_path=cache_path)


def _load_cache(*, cache_path: Path) -> Dict[str, Any]:
    if not cache_path.exists():
        return {}
    try:
        with cache_path.open("r", encoding="utf-8") as fh:
            data = json.load(fh)
            return data if isinstance(data, dict) else {}
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("Unable to read title normalization cache: %s", exc)
        return {}


def _save_cache(cache: Dict[str, Any], *, cache_path: Path) -> None:
    try:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with cache_path.open("w", encoding="utf-8") as fh:
            json.dump(cache, fh, ensure_ascii=False, indent=2)
    except OSError as exc:
        logger.warning("Unable to write title normalization cache: %s", exc)


def _cache_key(payload: Dict[str, Any]) -> str:
    serialized = json.dumps(payload, ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()
