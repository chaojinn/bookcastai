from __future__ import annotations

from typing import TYPE_CHECKING, Callable, List

try:  # pragma: no cover - import shim for script execution
    from ..epub_mcp import EbooklibEPUBMCPClient, TableOfContentsEntry
except ImportError:  # pragma: no cover
    from epub_mcp import EbooklibEPUBMCPClient, TableOfContentsEntry  # type: ignore

if TYPE_CHECKING:  # pragma: no cover - typing aid
    from ..epub_agent import EPUBAgentState


def make_fetch_table_of_contents_node(
    mcp_client: EbooklibEPUBMCPClient,
) -> Callable[["EPUBAgentState"], "EPUBAgentState"]:
    def fetch_table_of_contents(state: "EPUBAgentState") -> "EPUBAgentState":
        epub_path = state["epub_path"]
        sources = ("opf", "ncx")
        errors: List[str] = list(state.get("errors", []))
        for source in sources:
            response = mcp_client.get_table_of_contents(epub_path, source=source)
            if not response:
                errors.append(f"MCP client returned no response for TOC source '{source}'.")
                continue

            chapters = response.get("chapters") or []
            if chapters:
                toc_entries: List[TableOfContentsEntry] = []
                for entry in chapters:
                    href = entry.get("href") or ""
                    anchor = entry.get("anchor")
                    toc_entries.append(
                        {
                            "title": entry.get("title", ""),
                            "href": href,
                            "anchor": anchor,
                        },
                    )
                return {
                    "toc_entries": toc_entries,
                    "toc_source": response.get("source", source),
                    "errors": errors,
                }

            errors.append(
                f"No chapters discovered when using TOC source '{source}'.",
            )

        errors.append("Unable to extract table of contents from either OPF or NCX manifest.")
        return {"toc_entries": [], "toc_source": "", "errors": errors}

    return fetch_table_of_contents
