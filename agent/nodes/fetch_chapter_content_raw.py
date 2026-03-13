from __future__ import annotations

from typing import TYPE_CHECKING, Callable, List

try:  # pragma: no cover - import shim for script execution
    from ..epub_mcp import EbooklibEPUBMCPClient
except ImportError:  # pragma: no cover
    from epub_mcp import EbooklibEPUBMCPClient  # type: ignore

if TYPE_CHECKING:  # pragma: no cover - typing aid
    from ..epub_agent import EPUBAgentState, ChapterPayload


def make_fetch_chapter_content_raw_node(
    mcp_client: EbooklibEPUBMCPClient,
) -> Callable[["EPUBAgentState"], "EPUBAgentState"]:
    def fetch_chapter_content_raw(state: "EPUBAgentState") -> "EPUBAgentState":
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

            chapters.append(
                {
                    "chapter_number": number,
                    "chapter_title": entry.get("title", ""),
                    "content_text": response.get("content_text", ""),
                    "href": href,
                },
            )
        return {"chapters": chapters, "errors": errors}

    return fetch_chapter_content_raw
