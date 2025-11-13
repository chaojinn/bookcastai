from __future__ import annotations

import hashlib
import json
import logging
import os
import re
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Iterable, List, Optional, Sequence

from dotenv import load_dotenv
from openai import OpenAI

if TYPE_CHECKING:  # pragma: no cover - typing aid
    from ..epub_agent import EPUBAgentState

logger = logging.getLogger(__name__)

_SENTENCE_SPLIT_PATTERN = re.compile(r"(?<=[.!?])\s+|\n+")
_MODEL_NAME = "tngtech/deepseek-r1t2-chimera:free"
_CACHE_PATH = Path("data") / "openroute_cache.json"
_AI_CHUNK_PROMPT_TEMPLATE = (
    "convert the following text from ocr so it can be feed to a tts software. using rules \n"
    "1) add extra space if ocr missed space result in 2 words combined without space. \n"
    "2) remove unneeded spaces from ocr. \n"
    "3) remove parts might be introduced to text by ocr process which make sentence break. \n"
    "4) make sure the text have proper uppercase / lowercase.\n"
    "5)make minimal changes to the text. \n"
    "6)return changed text in whole without adding anything at the front or end.\n"
    "here is the text: {chunk_text}"
)


def _normalize_whitespace(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip()


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
        return _normalize_whitespace(cleaned), 1
    combined = " ".join(collected)
    return _normalize_whitespace(combined), sentence_count


class EPUBTextProcessor:
    def __init__(self, *, ai_extract_text: bool = False) -> None:
        self._ai_extract_text = bool(ai_extract_text)
        self._openrouter_client: Optional[OpenAI] = None

    def normalize_titles(
        self,
        state: "EPUBAgentState",
        errors: List[str],
    ) -> Dict[str, Any]:
        chapters = state.get("chapters")
        chapter_list = chapters if isinstance(chapters, list) else []
        filtered = self.filter_chapters(chapter_list)
        normalized = self._normalize_titles_with_openroute(filtered, errors)
        return {"chapters": normalized}

    def normalize_first_sentences(
        self,
        chapters: Sequence[Dict[str, Any]],
        errors: List[str],
    ) -> Dict[str, Any]:
        chapter_payload: List[Dict[str, Any]] = []
        for chapter in chapters:
            if not isinstance(chapter, dict):
                chapter_payload.append(
                    {
                        "chapter_title": "",
                        "first_sentence": "",
                        "sentence_count": 0,
                    },
                )
                continue
            title = chapter.get("chapter_title")
            content_text = chapter.get("content_text") if isinstance(chapter.get("content_text"), str) else ""
            first_sentence, sentence_count = _extract_first_sentence(content_text)
            chapter_payload.append(
                {
                    "chapter_title": title if isinstance(title, str) else "",
                    "first_sentence": first_sentence,
                    "sentence_count": sentence_count,
                },
            )

        try:
            intro_json = normalize_first_sentences_with_openroute(chapter_payload)
            normalized_intros = json.loads(intro_json)
            if isinstance(normalized_intros, list):
                for chapter, intro, payload in zip(chapters, normalized_intros, chapter_payload):
                    if not isinstance(chapter, dict) or not isinstance(intro, str):
                        continue
                    sentence_count = payload.get("sentence_count") if isinstance(payload, dict) else 0
                    chapter_number = payload.get("chapter_number") if isinstance(payload, dict) else None
                    chapter_title_value = payload.get("chapter_title") if isinstance(payload, dict) else ""
                    title = chapter_title_value.strip() if isinstance(chapter_title_value, str) else ""
                    sentence_count = sentence_count if isinstance(sentence_count, int) and sentence_count > 0 else 1
                    chapter_number = chapter_number if isinstance(chapter_number, int) else None

                    content_text = chapter.get("content_text")
                    content_text = content_text if isinstance(content_text, str) else ""
                    intro_text = intro.strip()

                    content_has_title = bool(title) and title.lower() in content_text.lower()
                    if chapter_number is not None:
                        prefix = f"Chapter {chapter_number}"
                        if content_has_title:
                            if not intro_text.lower().startswith(prefix.lower()):
                                intro_text = f"{prefix} {intro_text}".strip() if intro_text else prefix
                        elif title:
                            prefix_with_title = f"{prefix}: {title}"
                            if not intro_text.lower().startswith(prefix_with_title.lower()):
                                if intro_text:
                                    intro_text = f"{prefix_with_title}. {intro_text}".strip()
                                else:
                                    intro_text = prefix_with_title
                        elif not intro_text:
                            intro_text = prefix

                    if content_text:
                        segments = _SENTENCE_SPLIT_PATTERN.split(content_text.strip())
                        remaining_segments = segments[sentence_count:] if sentence_count < len(segments) else []
                        remaining = [segment for segment in remaining_segments if segment]
                        if remaining:
                            chapter["content_text"] = f"{intro_text} " + " ".join(remaining)
                        else:
                            chapter["content_text"] = intro_text
                    else:
                        chapter["content_text"] = intro_text
        except Exception as exc:  # pragma: no cover - depends on external service
            logger.error("OpenRoute first sentence normalization failed: %s", exc)
            errors.append(f"OpenRoute first sentence normalization failed: {exc}")

        return {"chapters": chapters}

    def filter_chapters(
        self,
        chapters: Sequence[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        filtered: List[Dict[str, Any]] = []
        for chapter in chapters:
            if not isinstance(chapter, dict):
                continue
            chapter_number = chapter.get("chapter_number")
            if not isinstance(chapter_number, int):
                continue
            chapter_copy: Dict[str, Any] = dict(chapter)
            chapter_copy["chapter_number"] = len(filtered) + 1
            filtered.append(chapter_copy)
        return filtered

    def convert_chunks_for_tts(
        self,
        chapter: Optional[Dict[str, Any]],
        chunks: Sequence[str],
    ) -> List[str]:
        if not chunks:
            return []
        chunk_list = [chunk for chunk in chunks]
        if not self._ai_extract_text:
            return chunk_list

        try:
            client = self._get_openrouter_client()
        except RuntimeError as exc:
            logger.error("AI chunk extraction unavailable: %s", exc)
            return chunk_list

        chapter_title = ""
        chapter_number = None
        if isinstance(chapter, dict):
            chapter_title = chapter.get("chapter_title", "")
            chapter_number = chapter.get("chapter_number")
        label = chapter_title or "Chapter"
        if isinstance(chapter_number, int):
            label = f"Chapter {chapter_number}: {label}" if chapter_title else f"Chapter {chapter_number}"

        cleaned_chunks: List[str] = []
        total = len(chunk_list)
        for index, chunk in enumerate(chunk_list, start=1):
            logger.info("AI extracting chunk %d/%d for %s", index, total, label)
            if not isinstance(chunk, str) or not chunk.strip():
                cleaned_chunks.append(chunk)
                continue
            try:
                cleaned_chunk = self._clean_chunk_with_ai(client, chunk)
            except Exception as exc:  # pragma: no cover - network failure path
                logger.error(
                    "AI extraction failed for chunk %d/%d (%s): %s",
                    index,
                    total,
                    label,
                    exc,
                )
                cleaned_chunk = chunk
            cleaned_chunks.append(cleaned_chunk)
        return cleaned_chunks

    def _get_openrouter_client(self) -> OpenAI:
        if self._openrouter_client is not None:
            return self._openrouter_client
        load_dotenv()
        api_key = os.getenv("OPENROUTE_API_KEY")
        if not api_key:
            raise RuntimeError("OPENROUTE_API_KEY is not set. Unable to run AI chunk extraction.")
        self._openrouter_client = OpenAI(
            api_key=api_key,
            base_url="https://openrouter.ai/api/v1",
        )
        return self._openrouter_client

    def _clean_chunk_with_ai(self, client: OpenAI, chunk_text: str) -> str:
        prompt = _AI_CHUNK_PROMPT_TEMPLATE.format(chunk_text=chunk_text)
        response = client.chat.completions.create(
            model=_MODEL_NAME,
            temperature=0,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You clean OCR text so it can be consumed by TTS engines. Respond with the fixed text only."
                    ),
                },
                {
                    "role": "user",
                    "content": prompt,
                },
            ],
        )
        content = response.choices[0].message.content if response.choices else None
        if not content:
            logger.error("AI extraction returned empty response. Falling back to original chunk.")
            return chunk_text
        return content.strip()

    def _normalize_titles_with_openroute(
        self,
        chapters: List[Dict[str, Any]],
        errors: List[str],
    ) -> List[Dict[str, Any]]:
        if not chapters:
            return chapters

        titles = [
            chapter.get("chapter_title", "") if isinstance(chapter.get("chapter_title"), str) else ""
            for chapter in chapters
        ]

        try:
            normalized_json = convert_titles_with_openroute(titles)
            normalized_titles = json.loads(normalized_json)
        except Exception as exc:  # pragma: no cover - depends on external service
            logger.error("OpenRoute normalization failed: %s", exc)
            errors.append(f"OpenRoute chapter title normalization failed: {exc}")
            return chapters

        if not isinstance(normalized_titles, list) or len(normalized_titles) != len(chapters):
            logger.error(
                "OpenRoute normalization returned unexpected results. Expected %d titles, got: %s",
                len(chapters),
                normalized_titles,
            )
            errors.append("OpenRoute chapter title normalization returned unexpected results.")
            return chapters

        normalized_start_count = sum(
            1
            for title in normalized_titles
            if isinstance(title, str) and title.strip().lower().startswith("chapter")
        )
        if normalized_start_count < len(normalized_titles) * 0.5:
            logger.error(
                "OpenRoute normalization produced too few chapter-prefixed titles (%d of %d).",
                normalized_start_count,
                len(normalized_titles),
            )
            raise RuntimeError(
                "OpenRoute normalization produced titles that do not start with 'Chapter' frequently enough.",
            )

        normalized: List[Dict[str, Any]] = []
        for chapter, title in zip(chapters, normalized_titles):
            if not isinstance(title, str):
                logger.error("OpenRoute normalization produced a non-string title: %s", title)
                errors.append("OpenRoute normalization produced a non-string title; keeping original titles.")
                return chapters
            chapter_copy = dict(chapter)
            chapter_copy["chapter_title"] = title
            normalized.append(chapter_copy)
        return normalized


def convert_titles_with_openroute(titles: Sequence[str]) -> str:
    if not titles:
        return json.dumps([], ensure_ascii=False)

    payload = {
        "instructions": (
            "Normalize each chapter title so that any spelled-out or Roman numeral chapter numbers "
            "are converted to digits (e.g., 'Chapter One' -> 'Chapter 1', 'Chapter III' -> 'Chapter 3', '8 Chapter title' -> 'Chapter 8: Chapter title'). "
            "If a title does not already contain a chapter number, prepend the correct sequential chapter "
            "number using the format 'Chapter X: ' where X starts at 1 for the first title. Preserve the rest "
            "of the title wording after normalizing the number. Return the final list of titles as a JSON array "
            "of strings in the same order as provided. "
            "don't convert prolog, prologue , preface , introduction, foreword, preamble,  prelude, epilog, epilogue, conclusion, coda, afterword, postscript, finale chapters, return these as they are."
        ),
        "titles": list(titles),
    }

    messages = [
        {
            "role": "system",
            "content": (
                "You transform chapter titles. Output MUST be a JSON array of strings, nothing else."
            ),
        },
        {
            "role": "user",
            "content": json.dumps(payload, ensure_ascii=False),
        },
    ]

    request_payload = {
        "model": _MODEL_NAME,
        "temperature": 0,
        "messages": messages,
    }

    content = _get_cached_openroute_response(request_payload)
    if content is None:
        load_dotenv()
        api_key = os.getenv("OPENROUTE_API_KEY")
        if not api_key:
            logger.error("OPENROUTE_API_KEY is not set. Unable to normalize chapter titles via OpenRoute.")
            raise RuntimeError("OPENROUTE_API_KEY is not set in the environment or .env file.")

        client = OpenAI(
            api_key=api_key,
            base_url="https://openrouter.ai/api/v1",
        )

        response = client.chat.completions.create(**request_payload)
        content = response.choices[0].message.content if response.choices else None
        if not content:
            logger.error("OpenRoute API returned empty content for payload: %s", payload)
            raise RuntimeError("OpenRoute API returned no content for the request.")
        _store_openroute_response(request_payload, content)
    else:
        logger.info("OpenRoute title normalization cache hit.")

    try:
        normalized_titles = json.loads(content)
        logger.info("OpenRoute normalization output: %s", normalized_titles)
    except json.JSONDecodeError as exc:
        logger.error("OpenRoute response not valid JSON. Content: %s", content)
        raise RuntimeError("OpenRoute API response was not valid JSON.") from exc

    if not isinstance(normalized_titles, list) or any(
        not isinstance(item, str) for item in normalized_titles
    ):
        logger.error(
            "OpenRoute response did not contain JSON array of strings. Content: %s",
            normalized_titles,
        )
        raise RuntimeError("OpenRoute API response did not contain a JSON array of strings.")

    return json.dumps(normalized_titles, ensure_ascii=False)


def normalize_first_sentences_with_openroute(
    chapter_requests: Sequence[Dict[str, Any]],
) -> str:
    if not chapter_requests:
        return json.dumps([], ensure_ascii=False)

    chapter_payload: List[Dict[str, Any]] = []
    for request in chapter_requests:
        if not isinstance(request, dict):
            chapter_payload.append(
                {
                    "chapter_title": "",
                    "first_sentence": "",
                    "sentence_count": 0,
                },
            )
            continue
        first_sentence = request.get("first_sentence")
        chapter_payload.append(
            {
                "chapter_title": request.get("chapter_title", ""),
                "first_sentence": first_sentence if isinstance(first_sentence, str) else "",
                "sentence_count": request.get("sentence_count", 0),
            },
        )

    logger.info(
        "OpenRoute first sentence normalization input (%d chapters).",
        len(chapter_payload),
    )

    instructions = (
        "Extract chapter number and chapter title from 'chapter_title' field. Use them to normalize the first sentence of each chapter. "
        "If 'chapter_title' contains chapter number, first stence should be {chapter title}. content"
        "If not, it should be Chapter {chapter number}: {chapter title}. content"
        'Any dupplication of chapter title or number at the beginning of the sentence content should be removed.'
        "The rest should remain unchanged."
        "Return the final list of sentences as a JSON array of strings in the same order as provided."
    )

    messages = [
        {
            "role": "system",
            "content": (
                "You normalize chapter openings. Output MUST be a JSON array of strings, nothing else."
            ),
        },
        {
            "role": "user",
            "content": json.dumps(
                {
                    "instructions": instructions,
                    "chapters": chapter_payload,
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

    content = _get_cached_openroute_response(request_payload)
    if content is None:
        load_dotenv()
        api_key = os.getenv("OPENROUTE_API_KEY")
        if not api_key:
            logger.error("OPENROUTE_API_KEY is not set. Unable to normalize first sentences.")
            raise RuntimeError("OPENROUTE_API_KEY is not set in the environment or .env file.")

        client = OpenAI(
            api_key=api_key,
            base_url="https://openrouter.ai/api/v1",
        )

        response = client.chat.completions.create(**request_payload)
        content = response.choices[0].message.content if response.choices else None
        if not content:
            logger.error("OpenRoute API returned empty content for introduction normalization.")
            raise RuntimeError("OpenRoute API returned no content for the request.")
        _store_openroute_response(request_payload, content)
    else:
        logger.info("OpenRoute first-sentence normalization cache hit.")

    try:
        normalized_sentences = json.loads(content)
        logger.info("OpenRoute first sentence normalization output: %s", normalized_sentences)
    except json.JSONDecodeError as exc:
        logger.error("OpenRoute response for first sentences not valid JSON. Content: %s", content)
        raise RuntimeError("OpenRoute API response was not valid JSON.") from exc

    if not isinstance(normalized_sentences, list) or len(normalized_sentences) != len(chapter_payload):
        logger.error(
            "OpenRoute first sentence normalization returned unexpected results. Expected %d entries, got: %s",
            len(chapter_payload),
            normalized_sentences,
        )
        raise RuntimeError("OpenRoute API response did not contain the expected number of sentences.")

    for normalized in normalized_sentences:
        if not isinstance(normalized, str):
            logger.error("OpenRoute normalization produced a non-string sentence: %s", normalized)
            raise RuntimeError("OpenRoute normalization produced a non-string first sentence.")

    return json.dumps(normalized_sentences, ensure_ascii=False)


def _get_cached_openroute_response(payload: Dict[str, Any]) -> str | None:
    cache = _load_openroute_cache()
    key = _openroute_cache_key(payload)
    entry = cache.get(key)
    if not isinstance(entry, dict):
        return None
    response = entry.get("response")
    return response if isinstance(response, str) else None


def _store_openroute_response(payload: Dict[str, Any], response: str) -> None:
    cache = _load_openroute_cache()
    key = _openroute_cache_key(payload)
    cache[key] = {"request": payload, "response": response}
    _save_openroute_cache(cache)


def _load_openroute_cache() -> Dict[str, Any]:
    if not _CACHE_PATH.exists():
        return {}
    try:
        with _CACHE_PATH.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
            return data if isinstance(data, dict) else {}
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("Unable to read OpenRoute cache: %s", exc)
        return {}


def _save_openroute_cache(cache: Dict[str, Any]) -> None:
    try:
        _CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
        with _CACHE_PATH.open("w", encoding="utf-8") as handle:
            json.dump(cache, handle, ensure_ascii=False, indent=2)
    except OSError as exc:
        logger.warning("Unable to write OpenRoute cache: %s", exc)


def _openroute_cache_key(payload: Dict[str, Any]) -> str:
    serialized = json.dumps(payload, ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()
