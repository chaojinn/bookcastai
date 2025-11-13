from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable, Dict

if TYPE_CHECKING:  # pragma: no cover - typing aid
    from ..epub_agent import EPUBAgentState


def make_assemble_payload_node() -> Callable[["EPUBAgentState"], "EPUBAgentState"]:
    def assemble_payload(state: "EPUBAgentState") -> "EPUBAgentState":
        payload: Dict[str, Any] = dict(state.get("metadata", {}))
        chapters = state.get("chapters", [])
        if isinstance(chapters, list) and chapters:
            payload["chapters"] = [
                dict(chapter) for chapter in chapters if isinstance(chapter, dict)
            ]
        cover_image = state.get("cover_image")
        if cover_image:
            payload["cover_image"] = cover_image
        cover_type = state.get("cover_image_media_type")
        if cover_type:
            payload["cover_image_media_type"] = cover_type

        toc_source = state.get("toc_source")
        if toc_source:
            payload["toc_source"] = toc_source

        errors = state.get("errors", [])
        if errors:
            payload["_warnings"] = errors
        return {"result": payload}

    return assemble_payload
