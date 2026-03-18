from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional

try:
    from ._extract_utils import (
        _ContentStripper,
        _generate_cleanup_rules,
    )
    from ..epub_mcp import EbooklibEPUBMCPClient
except ImportError:
    from _extract_utils import _ContentStripper, _generate_cleanup_rules  # type: ignore
    from epub_mcp import EbooklibEPUBMCPClient  # type: ignore

if TYPE_CHECKING:
    from ..epub_agent import EPUBAgentState

logger = logging.getLogger(__name__)

_PREVIEW_MAX_CHARS = 300


def _compute_rule_preview(
    rule: str,
    raw_html_map: Dict[str, str],
    stripper: _ContentStripper,
) -> str:
    """Return a plain-text preview of the first content removed by *rule* across all chapters."""
    for chapter_number in sorted(raw_html_map, key=lambda k: int(k) if k.isdigit() else k):
        html = raw_html_map[chapter_number]
        try:
            matches = re.findall(rule, html, flags=re.DOTALL | re.IGNORECASE)
        except re.error:
            return ""
        for m in matches:
            if not isinstance(m, str):
                continue
            text = stripper.clean(m)
            if text:
                return text[:_PREVIEW_MAX_CHARS]
    return ""


def make_generate_rules_node(
    mcp_client: EbooklibEPUBMCPClient,
    *,
    cache_path: Optional[Path] = None,
) -> Callable[["EPUBAgentState"], "EPUBAgentState"]:
    stripper = _ContentStripper()

    def generate_rules(state: "EPUBAgentState") -> "EPUBAgentState":
        chapters: List[Dict[str, Any]] = list(state.get("chapters") or [])
        epub_path: str = state.get("epub_path", "")
        errors: List[str] = list(state.get("errors") or [])

        if not chapters or not epub_path:
            return {"chapters": chapters, "errors": errors, "cleanup_rules": [], "raw_html_map": {}, "rule_previews": []}

        # Fetch raw HTML for all chapters (use str keys — int keys are lost by JSON serialization in checkpointer)
        raw_html_map: Dict[str, str] = {}
        for chapter in chapters:
            href = chapter.get("href", "")
            chapter_number = chapter.get("chapter_number")
            if not href or not isinstance(chapter_number, int):
                continue
            response = mcp_client.get_chapter_content(epub_path, href=href)
            if response:
                raw_html_map[str(chapter_number)] = response.get("content_text", "")

        if not raw_html_map:
            logger.warning("No raw HTML could be fetched; skipping rule generation.")
            return {"chapters": chapters, "errors": errors, "cleanup_rules": [], "raw_html_map": {}, "rule_previews": []}

        # Pick representative chapter (median by length)
        sorted_by_length = sorted(raw_html_map.items(), key=lambda x: len(x[1]))
        median_idx = len(sorted_by_length) // 2
        representative_number, representative_html = sorted_by_length[median_idx]
        logger.info(
            "Representative chapter for rule generation: chapter_number=%s, html_length=%s",
            representative_number,
            len(representative_html),
        )

        # Ask LLM to generate regex rules
        rules = _generate_cleanup_rules(representative_html, errors, cache_path=cache_path)
        if not rules:
            logger.warning("No cleanup rules generated.")

        # Compute a plain-text preview for each rule
        rule_previews: List[Dict[str, Any]] = [
            {"rule": rule, "preview": _compute_rule_preview(rule, raw_html_map, stripper)}
            for rule in rules
        ]

        return {
            "raw_html_map": raw_html_map,
            "cleanup_rules": rules,
            "rule_previews": rule_previews,
            "errors": errors,
        }

    return generate_rules
