from __future__ import annotations

import hashlib
import json
import logging
import os
import re
from html.parser import HTMLParser
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional

from dotenv import load_dotenv
from openai import OpenAI

try:  # pragma: no cover - import shim for script execution
    from ..epub_mcp import EbooklibEPUBMCPClient
except ImportError:  # pragma: no cover
    from epub_mcp import EbooklibEPUBMCPClient  # type: ignore

if TYPE_CHECKING:  # pragma: no cover - typing aid
    from ..epub_agent import EPUBAgentState

logger = logging.getLogger(__name__)

_MODEL_NAME = "openai/gpt-5.2"

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

    def handle_starttag(self, tag: str, attrs: object) -> None:  # noqa: ARG002
        if tag.lower() in _BLOCK_LEVEL_TAGS:
            self._append_separator()

    def handle_startendtag(self, tag: str, attrs: object) -> None:  # noqa: ARG002
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


def make_extract_text_node(
    mcp_client: EbooklibEPUBMCPClient,
    *,
    cache_path: Optional[Path] = None,
    debug_output_path: Optional[Path] = None,
) -> Callable[["EPUBAgentState"], "EPUBAgentState"]:
    stripper = _ContentStripper()

    def extract_text(state: "EPUBAgentState") -> "EPUBAgentState":
        chapters: List[Dict[str, Any]] = list(state.get("chapters") or [])
        epub_path: str = state.get("epub_path", "")
        errors: List[str] = list(state.get("errors") or [])

        if not chapters or not epub_path:
            return {"chapters": chapters, "errors": errors}

        # Fetch raw HTML for all chapters
        raw_html_map: Dict[int, str] = {}
        for chapter in chapters:
            href = chapter.get("href", "")
            chapter_number = chapter.get("chapter_number")
            if not href or not isinstance(chapter_number, int):
                continue
            response = mcp_client.get_chapter_content(epub_path, href=href)
            if response:
                raw_html_map[chapter_number] = response.get("content_text", "")

        if not raw_html_map:
            logger.warning("No raw HTML could be fetched; skipping extract_text.")
            return {"chapters": chapters, "errors": errors}

        # Sort by raw HTML length and pick chapter closest to the median
        sorted_by_length = sorted(raw_html_map.items(), key=lambda x: len(x[1]))
        median_idx = len(sorted_by_length) // 2
        representative_number, representative_html = sorted_by_length[median_idx]
        logger.info(
            "Representative chapter for rule generation: chapter_number=%s, html_length=%s",
            representative_number,
            len(representative_html),
        )

        # Ask LLM to generate regex rules for removing unwanted content
        rules = _generate_cleanup_rules(representative_html, errors, cache_path=cache_path)
        if not rules:
            logger.warning("No cleanup rules generated; using raw HTML with plain stripping.")

        # Apply rules to each chapter's raw HTML, then strip
        updated_chapters = []
        removed_text_map: Dict[int, List[str]] = {}
        for chapter in chapters:
            chapter_number = chapter.get("chapter_number")
            raw_html = raw_html_map.get(chapter_number, "") if isinstance(chapter_number, int) else ""
            if raw_html:
                cleaned_html, removed = _apply_rules(raw_html, rules, stripper)
                content_text = stripper.clean(cleaned_html)
                if isinstance(chapter_number, int):
                    removed_text_map[chapter_number] = removed
            else:
                content_text = chapter.get("content_text", "")
            chapter_copy = dict(chapter)
            chapter_copy["content_text"] = content_text
            updated_chapters.append(chapter_copy)

        if debug_output_path is not None:
            _save_debug(updated_chapters, rules, removed_text_map, debug_output_path)

        return {"chapters": updated_chapters, "errors": errors}

    return extract_text


def _apply_rules(html: str, rules: List[str], stripper: _ContentStripper) -> tuple[str, List[str]]:
    removed: List[str] = []
    for pattern in rules:
        try:
            matches = re.findall(pattern, html, flags=re.DOTALL | re.IGNORECASE)
            for m in matches:
                text = stripper.clean(m) if isinstance(m, str) else ""
                if text:
                    removed.append(text)
            html = re.sub(pattern, "", html, flags=re.DOTALL | re.IGNORECASE)
        except re.error as exc:
            logger.warning("Skipping invalid regex rule '%s': %s", pattern, exc)
    return html, removed


def _generate_cleanup_rules(
    html_sample: str,
    errors: List[str],
    *,
    cache_path: Optional[Path],
) -> List[str]:
    instructions = (
        "You are analyzing an HTML chapter from an epub book to identify unwanted content "
        "that should be removed before generating an audio book.\n"
        "Unwanted content includes: tables, image captions, footnotes, page numbers, "
        "synopsis, chapter summaries, sidebars, and any non-narrative elements.\n"
        "Generate regex patterns that match HTML elements containing such content. "
        "Each pattern must match a complete HTML fragment (opening tag through closing tag or self-closing). "
        "Use tag names, class name patterns, or id patterns to target the right elements. "
        "Patterns will be applied with re.sub(pattern, '', html, flags=re.DOTALL|re.IGNORECASE).\n"
        "Return a JSON array of regex pattern strings, nothing else. "
        "If no unwanted content is detected, return an empty array []."
    )

    messages = [
        {
            "role": "system",
            "content": "Output MUST be a JSON array of regex strings, nothing else.",
        },
        {
            "role": "user",
            "content": json.dumps(
                {"instructions": instructions, "html_sample": html_sample},
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
            logger.info("Cleanup rules cache hit.")
            content = cached

    if content is None:
        load_dotenv()
        api_key = os.getenv("OPENROUTE_API_KEY")
        if not api_key:
            msg = "OPENROUTE_API_KEY is not set; skipping cleanup rule generation."
            logger.warning(msg)
            errors.append(msg)
            return []

        client = OpenAI(api_key=api_key, base_url="https://openrouter.ai/api/v1")
        try:
            response = client.chat.completions.create(**request_payload)
        except Exception as exc:  # pragma: no cover - depends on network
            logger.error("Cleanup rule generation LLM call failed: %s", exc)
            errors.append(f"Cleanup rule generation failed: {exc}")
            return []

        content = response.choices[0].message.content if response.choices else None
        if not content:
            logger.error("Cleanup rule generation returned empty content.")
            errors.append("Cleanup rule generation returned empty content.")
            return []

        if cache_path is not None:
            _store_cached_response(request_payload, content, cache_path=cache_path)

    try:
        raw_rules = json.loads(content)
    except json.JSONDecodeError:
        logger.error("Cleanup rules response is not valid JSON: %s", content)
        errors.append("Cleanup rules response is not valid JSON.")
        return []

    if not isinstance(raw_rules, list):
        logger.error("Cleanup rules response is not a list: %s", raw_rules)
        errors.append("Cleanup rules response is not a list.")
        return []

    # Validate each rule is a valid regex
    valid_rules: List[str] = []
    for rule in raw_rules:
        if not isinstance(rule, str):
            continue
        try:
            re.compile(rule, re.DOTALL | re.IGNORECASE)
            valid_rules.append(rule)
        except re.error as exc:
            logger.warning("Discarding invalid regex rule '%s': %s", rule, exc)
            errors.append(f"Discarding invalid regex rule: {rule!r} ({exc})")

    logger.info("Generated %d valid cleanup rules.", len(valid_rules))
    return valid_rules


def _save_debug(
    chapters: List[Dict[str, Any]],
    rules: List[str],
    removed_text_map: Dict[int, List[str]],
    output_path: Path,
) -> None:
    payload = {
        "rules": rules,
        "chapters": [
            {
                "chapter_number": ch.get("chapter_number"),
                "chapter_title": ch.get("chapter_title", ""),
                "chapter_content": ch.get("content_text", ""),
                "removed_text": removed_text_map.get(ch.get("chapter_number"), []),
            }
            for ch in chapters
        ],
    }
    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2, ensure_ascii=False)
        logger.info("Cleaned debug output written to %s", output_path)
    except OSError as exc:
        logger.warning("Failed to write cleaned debug output: %s", exc)


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
        logger.warning("Unable to read cleanup rules cache: %s", exc)
        return {}


def _save_cache(cache: Dict[str, Any], *, cache_path: Path) -> None:
    try:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with cache_path.open("w", encoding="utf-8") as fh:
            json.dump(cache, fh, ensure_ascii=False, indent=2)
    except OSError as exc:
        logger.warning("Unable to write cleanup rules cache: %s", exc)


def _cache_key(payload: Dict[str, Any]) -> str:
    serialized = json.dumps(payload, ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()
