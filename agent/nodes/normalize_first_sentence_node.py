from __future__ import annotations

from typing import TYPE_CHECKING, Callable, List

from .text_processing import EPUBTextProcessor

if TYPE_CHECKING:  # pragma: no cover
    from ..epub_agent import EPUBAgentState


def make_normalize_first_sentence_node(
    processor: EPUBTextProcessor,
) -> Callable[["EPUBAgentState"], "EPUBAgentState"]:
    def normalize_first_sentence(state: "EPUBAgentState") -> "EPUBAgentState":
        errors: List[str] = list(state.get("errors", []))
        result = processor.normalize_first_sentences(state, errors)
        if errors:
            result["errors"] = errors
        return result

    return normalize_first_sentence
