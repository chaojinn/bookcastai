from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional

try:
    from ._extract_utils import _ContentStripper, _apply_rules, _save_debug
except ImportError:
    from _extract_utils import _ContentStripper, _apply_rules, _save_debug  # type: ignore

if TYPE_CHECKING:
    from ..epub_agent import EPUBAgentState

logger = logging.getLogger(__name__)


def make_apply_rules_node(
    *,
    debug_output_path: Optional[Path] = None,
) -> Callable[["EPUBAgentState"], "EPUBAgentState"]:
    stripper = _ContentStripper()

    def apply_rules(state: "EPUBAgentState") -> "EPUBAgentState":
        chapters: List[Dict[str, Any]] = list(state.get("chapters") or [])
        approved_rules: List[str] = list(state.get("approved_rules") or [])
        raw_html_map: Dict[str, str] = dict(state.get("raw_html_map") or {})
        errors: List[str] = list(state.get("errors") or [])

        updated_chapters = []
        removed_text_map: Dict[int, List[str]] = {}

        for chapter in chapters:
            chapter_number = chapter.get("chapter_number")
            # Use str key — int keys are lost when the checkpointer serializes state via JSON
            raw_html = raw_html_map.get(str(chapter_number), "") if isinstance(chapter_number, int) else ""
            if raw_html:
                if approved_rules:
                    cleaned_html, removed = _apply_rules(raw_html, approved_rules, stripper)
                else:
                    cleaned_html, removed = raw_html, []
                content_text = stripper.clean(cleaned_html)
                if isinstance(chapter_number, int):
                    removed_text_map[chapter_number] = removed
            else:
                content_text = chapter.get("content_text", "")

            chapter_copy = dict(chapter)
            chapter_copy["content_text"] = content_text
            updated_chapters.append(chapter_copy)

        if debug_output_path is not None:
            _save_debug(updated_chapters, approved_rules, removed_text_map, debug_output_path)

        return {"chapters": updated_chapters, "errors": errors}

    return apply_rules
