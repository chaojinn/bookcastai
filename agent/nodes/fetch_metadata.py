from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional

try:  # pragma: no cover - import shim for script execution
    from ..epub_mcp import EbooklibEPUBMCPClient
except ImportError:  # pragma: no cover
    from epub_mcp import EbooklibEPUBMCPClient  # type: ignore

if TYPE_CHECKING:  # pragma: no cover - typing aid
    from ..epub_agent import EPUBAgentState


def make_fetch_metadata_node(
    mcp_client: EbooklibEPUBMCPClient,
) -> Callable[["EPUBAgentState"], "EPUBAgentState"]:
    def fetch_metadata(state: "EPUBAgentState") -> "EPUBAgentState":
        epub_path = state["epub_path"]
        errors: List[str] = list(state.get("errors", []))

        response = mcp_client.get_metadata(epub_path)
        if not response:
            errors.append("MCP client returned no metadata for the EPUB file.")
            return {
                "metadata": {},
                "cover_image": None,
                "cover_image_media_type": None,
                "errors": errors,
            }

        metadata: Dict[str, Any] = response.get("metadata") or {}
        cover_image: Optional[str] = response.get("cover_image")
        cover_type: Optional[str] = response.get("cover_image_media_type")
        return {
            "metadata": metadata,
            "cover_image": cover_image,
            "cover_image_media_type": cover_type,
            "errors": errors,
        }

    return fetch_metadata
