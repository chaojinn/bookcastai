from __future__ import annotations

import logging
import re
from html.parser import HTMLParser
from typing import TYPE_CHECKING, Callable, List, Optional

try:  # pragma: no cover - import shim for script execution
    from ..epub_mcp import EbooklibEPUBMCPClient
except ImportError:  # pragma: no cover
    from epub_mcp import EbooklibEPUBMCPClient  # type: ignore

if TYPE_CHECKING:  # pragma: no cover - typing aid
    from ..epub_agent import EPUBAgentState, ChapterPayload


def make_fetch_chapter_content_node(
    mcp_client: EbooklibEPUBMCPClient,
    *,
    ignore_classes: Optional[List[str]] = None,
) -> Callable[["EPUBAgentState"], "EPUBAgentState"]:
    stripper = _ContentStripper(ignore_classes or [])

    def fetch_chapter_content(state: "EPUBAgentState") -> "EPUBAgentState":
        toc_entries = state.get("toc_entries", [])
        epub_path = state["epub_path"]
        errors: List[str] = list(state.get("errors", []))
        if not toc_entries:
            errors.append("Chapter extraction skipped because TOC entries are missing.")
            return {"chapters": [], "errors": errors}

        chapters: List["ChapterPayload"] = []
        for number, entry in enumerate(toc_entries, start=1):
            href = entry.get("href", "")
            if not href:
                errors.append(f"Skipping chapter {number}: missing href in TOC entry.")
                continue
            response = mcp_client.get_chapter_content(epub_path, href=href)
            if not response:
                errors.append(f"MCP client returned no content for chapter href '{href}'.")
                continue

            raw_content = response.get("content_text", "")
            content_text = stripper.clean(raw_content)
            chapters.append(
                {
                    "chapter_number": number,
                    "chapter_title": entry.get("title", ""),
                    "content_text": content_text,
                },
            )
        return {"chapters": chapters, "errors": errors}

    return fetch_chapter_content


logger = logging.getLogger(__name__)


_BLOCK_LEVEL_TAGS = {
    "article",
    "aside",
    "blockquote",
    "br",
    "div",
    "footer",
    "h1",
    "h2",
    "h3",
    "h4",
    "h5",
    "h6",
    "header",
    "hr",
    "li",
    "main",
    "nav",
    "p",
    "pre",
    "section",
    "ul",
    "ol",
}


class _ContentStripper(HTMLParser):
    """Strip HTML tags while honoring ignored classes."""

    def __init__(self, ignored_classes: List[str]) -> None:
        super().__init__()
        self._ignored_classes = {
            cls.strip().lower()
            for cls in ignored_classes
            if isinstance(cls, str) and cls.strip()
        }
        self._parts: List[str] = []
        self._ignore_stack: List[bool] = []

    def clean(self, html_text: str) -> str:
        if not isinstance(html_text, str):
            return ""
        self.reset()
        self._parts.clear()
        self._ignore_stack.clear()
        self.feed(html_text)
        self.close()
        text = "".join(self._parts).strip()
        text = text.replace("\n", " ")
        cleaned = " ".join(text.split())
        # Remove parenthetical content entirely.
        cleaned = re.sub(r"\([^)]*\)", "", cleaned)
        cleaned = " ".join(cleaned.split())
        return cleaned

    def handle_starttag(self, tag: str, attrs: List[tuple[str, Optional[str]]]) -> None:
        if tag.lower() in _BLOCK_LEVEL_TAGS:
            self._append_separator()
        self._push_ignore(attrs)

    def handle_startendtag(self, tag: str, attrs: List[tuple[str, Optional[str]]]) -> None:
        if tag.lower() in _BLOCK_LEVEL_TAGS:
            self._append_separator()
        self._push_ignore(attrs)
        self.handle_endtag(tag)

    def handle_endtag(self, tag: str) -> None:
        if tag.lower() in _BLOCK_LEVEL_TAGS:
            self._append_separator()
        if self._ignore_stack:
            self._ignore_stack.pop()

    def handle_data(self, data: str) -> None:
        if not data:
            return
        if self._ignore_stack and self._ignore_stack[-1]:
            return
        self._parts.append(data)

    def _push_ignore(self, attrs: List[tuple[str, Optional[str]]]) -> None:
        if self._ignore_stack and self._ignore_stack[-1]:
            self._ignore_stack.append(True)
            return
        should_ignore = False
        if self._ignored_classes:
            for name, value in attrs:
                if not name or name.lower() != "class" or not value:
                    continue
                classes = {segment.strip().lower() for segment in value.split() if segment.strip()}
                if classes & self._ignored_classes:
                    should_ignore = True
                    break
        self._ignore_stack.append(should_ignore)

    def _append_separator(self) -> None:
        if not self._parts:
            return
        last = self._parts[-1]
        if not last:
            return
        # Ensure block boundaries end sentences cleanly.
        last_non_space = None
        for ch in reversed(last):
            if not ch.isspace():
                last_non_space = ch
                break
        if last_non_space is None:
            return
        if last_non_space not in {".", "?", "!", ":", ";",","}:
            self._parts[-1] = last.rstrip() + "."
        self._parts.append(" ")
