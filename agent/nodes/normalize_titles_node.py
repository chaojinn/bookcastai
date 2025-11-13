from __future__ import annotations

from typing import TYPE_CHECKING, Callable, List

from .text_processing import EPUBTextProcessor

if TYPE_CHECKING:  # pragma: no cover
    from ..epub_agent import EPUBAgentState


def make_normalize_titles_node(
    processor: EPUBTextProcessor,
) -> Callable[["EPUBAgentState"], "EPUBAgentState"]:
    def normalize_titles(state: "EPUBAgentState") -> "EPUBAgentState":
        errors: List[str] = list(state.get("errors", []))
        result = processor.normalize_titles(state, errors)
        if errors:
            result["errors"] = errors
        return result

    return normalize_titles
