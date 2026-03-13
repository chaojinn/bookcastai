from __future__ import annotations

import hashlib
import json
import logging
import os
import re
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional

from dotenv import load_dotenv
from openai import OpenAI

if TYPE_CHECKING:  # pragma: no cover - typing aid
    from ..epub_agent import EPUBAgentState

logger = logging.getLogger(__name__)

_MODEL_NAME = "openai/gpt-5.2"
_SENTENCE_SPLIT_PATTERN = re.compile(r"(?<=[.!?])\s+|\n+")


def _extract_first_sentence(text: str) -> tuple[str, int]:
    if not isinstance(text, str):
        return "", 0
    cleaned = text.strip()
    if not cleaned:
        return "", 0
    parts = _SENTENCE_SPLIT_PATTERN.split(cleaned)
    collected: List[str] = []
    total_length = 0
    sentence_count = 0
    for segment in parts:
        if not segment:
            continue
        collected.append(segment)
        total_length += len(segment.strip())
        sentence_count += 1
        if total_length >= 50:
            break
    if not collected:
        return re.sub(r"\s+", " ", cleaned).strip(), 1
    combined = " ".join(collected)
    return re.sub(r"\s+", " ", combined).strip(), sentence_count


def _replace_first_sentence(content_text: str, sentence_count: int, new_first: str) -> str:
    segments = _SENTENCE_SPLIT_PATTERN.split(content_text.strip())
    remaining = [s for s in segments[sentence_count:] if s]
    if remaining:
        return f"{new_first} " + " ".join(remaining)
    return new_first


def make_normalize_first_sentence_node(
    cache_path: Optional[Path] = None,
    debug_output_path: Optional[Path] = None,
) -> Callable[["EPUBAgentState"], "EPUBAgentState"]:
    def normalize_first_sentence_new(state: "EPUBAgentState") -> "EPUBAgentState":
        chapters: List[Dict[str, Any]] = list(state.get("chapters") or [])
        errors: List[str] = list(state.get("errors") or [])

        if not chapters:
            return {"chapters": chapters, "errors": errors}

        # Extract first sentence per chapter
        first_sentences: List[tuple[str, int]] = [
            _extract_first_sentence(ch.get("content_text", "") if isinstance(ch.get("content_text"), str) else "")
            for ch in chapters
        ]

        # Build payload for LLM
        llm_input = [
            {
                "chapter_number": ch.get("chapter_number"),
                "chapter_title": ch.get("chapter_title", "") if isinstance(ch.get("chapter_title"), str) else "",
                "first_sentence": first_sentences[idx][0],
            }
            for idx, ch in enumerate(chapters)
        ]

        llm_results = _query_llm_for_duplicates(llm_input, errors, cache_path=cache_path)

        debug_records: List[Dict[str, Any]] = []
        updated_chapters: List[Dict[str, Any]] = []

        for idx, chapter in enumerate(chapters):
            title = chapter.get("chapter_title", "") if isinstance(chapter.get("chapter_title"), str) else ""
            content_text = chapter.get("content_text", "") if isinstance(chapter.get("content_text"), str) else ""
            old_first, sentence_count = first_sentences[idx]
            sentence_count = sentence_count if sentence_count > 0 else 1

            # Apply LLM result if changed
            new_first = ""
            if llm_results and idx < len(llm_results):
                result = llm_results[idx]
                if isinstance(result, dict) and result.get("changed") is True:
                    changed_text = result.get("changed_text", "")
                    if isinstance(changed_text, str) and changed_text.strip():
                        new_first = changed_text.strip()

            effective_first = new_first if new_first else old_first

            # Rebuild content: title + first sentence + rest
            if content_text:
                body = _replace_first_sentence(content_text, sentence_count, effective_first)
                new_content = f"{title}. {body}" if title else body
            else:
                new_content = f"{title}. {effective_first}" if title else effective_first

            debug_records.append({
                "chapter_title": title,
                "old_first_sentence": old_first,
                "new_first_sentence": new_first,
            })

            chapter_copy = dict(chapter)
            chapter_copy["content_text"] = new_content
            updated_chapters.append(chapter_copy)

        if debug_output_path is not None:
            _save_debug(debug_records, debug_output_path)

        return {"chapters": updated_chapters, "errors": errors}

    return normalize_first_sentence_new


def _query_llm_for_duplicates(
    llm_input: List[Dict[str, Any]],
    errors: List[str],
    *,
    cache_path: Optional[Path],
) -> Optional[List[Dict[str, Any]]]:
    if not llm_input:
        return None

    instructions = (
        "For each chapter, check if the chapter title is replicated in the first sentence in any form "
        "(e.g. title says 'Chapter 3' and sentence starts with 'chapter III' or 'III' or 'Chapter Three').\n"
        "If the title is duplicated, remove that duplicated part from the first sentence and return the cleaned sentence.\n"
        "Return a JSON array with one object per chapter in the same order:\n"
        '{"changed": true/false, "changed_text": "<cleaned first sentence or empty string if not changed>"}\n'
        "Return only the JSON array, nothing else."
    )

    messages = [
        {
            "role": "system",
            "content": "Output MUST be a JSON array of objects with 'changed' and 'changed_text' fields, nothing else.",
        },
        {
            "role": "user",
            "content": json.dumps(
                {"instructions": instructions, "chapters": llm_input},
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
            logger.info("First sentence normalization cache hit.")
            content = cached

    if content is None:
        load_dotenv()
        api_key = os.getenv("OPENROUTE_API_KEY")
        if not api_key:
            msg = "OPENROUTE_API_KEY is not set; skipping first sentence normalization."
            logger.warning(msg)
            errors.append(msg)
            return None

        client = OpenAI(api_key=api_key, base_url="https://openrouter.ai/api/v1")
        try:
            response = client.chat.completions.create(**request_payload)
        except Exception as exc:  # pragma: no cover - depends on network
            logger.error("First sentence normalization LLM call failed: %s", exc)
            errors.append(f"First sentence normalization LLM call failed: {exc}")
            return None

        content = response.choices[0].message.content if response.choices else None
        if not content:
            logger.error("First sentence normalization returned empty content.")
            errors.append("First sentence normalization returned empty content.")
            return None

        if cache_path is not None:
            _store_cached_response(request_payload, content, cache_path=cache_path)

    try:
        results = json.loads(content)
    except json.JSONDecodeError:
        logger.error("First sentence normalization returned invalid JSON: %s", content)
        errors.append("First sentence normalization returned invalid JSON.")
        return None

    if not isinstance(results, list) or len(results) != len(llm_input):
        logger.error(
            "First sentence normalization returned unexpected result (expected %d items): %s",
            len(llm_input),
            results,
        )
        errors.append("First sentence normalization returned unexpected result.")
        return None

    validated: List[Dict[str, Any]] = []
    for item in results:
        if not isinstance(item, dict) or "changed" not in item or "changed_text" not in item:
            logger.error("First sentence normalization item missing required fields: %s", item)
            errors.append("First sentence normalization item missing required fields.")
            return None
        validated.append(item)

    logger.info("First sentence normalization complete for %d chapters.", len(validated))
    return validated


def _save_debug(records: List[Dict[str, Any]], output_path: Path) -> None:
    payload = {
        "chapters": [
            {
                "chapter_title": r.get("chapter_title", ""),
                "old_first_sentence": r.get("old_first_sentence", ""),
                "new_first_sentence": r.get("new_first_sentence", ""),
            }
            for r in records
        ]
    }
    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2, ensure_ascii=False)
        logger.info("First sentence debug output written to %s", output_path)
    except OSError as exc:
        logger.warning("Failed to write first sentence debug output: %s", exc)


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
        logger.warning("Unable to read first sentence cache: %s", exc)
        return {}


def _save_cache(cache: Dict[str, Any], *, cache_path: Path) -> None:
    try:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with cache_path.open("w", encoding="utf-8") as fh:
            json.dump(cache, fh, ensure_ascii=False, indent=2)
    except OSError as exc:
        logger.warning("Unable to write first sentence cache: %s", exc)


def _cache_key(payload: Dict[str, Any]) -> str:
    serialized = json.dumps(payload, ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()
