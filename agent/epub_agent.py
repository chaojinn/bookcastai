from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, TypedDict

try:
    from langgraph.graph import END, StateGraph
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "langgraph is required. Install it via 'pip install langgraph'.",
    ) from exc


class ChapterPayload(TypedDict, total=False):
    """Normalized structure for chapter content."""

    chapter_number: int
    chapter_title: str
    content_text: str
    href: str
    chunks: List[str]


class EPUBAgentState(TypedDict, total=False):
    """State container shared across LangGraph nodes."""

    epub_path: str
    toc_entries: List[Dict[str, Any]]
    toc_source: str
    chapters: List[Dict[str, Any]]
    metadata: Dict[str, Any]
    cover_image: Optional[str]
    cover_image_media_type: Optional[str]
    errors: List[str]
    result: Dict[str, Any]


try:  # pragma: no cover - import shim for script execution
    from .epub_mcp import EbooklibEPUBMCPClient
    from .nodes.fetch_metadata import make_fetch_metadata_node
    from .nodes.fetch_table_of_contents import make_fetch_table_of_contents_node
    from .nodes.fetch_chapter_content_raw import make_fetch_chapter_content_raw_node
    from .nodes.construct_book_structure import make_construct_book_structure_node
    from .nodes.normalize_titles import make_normalize_titles_node
    from .nodes.extract_text import make_extract_text_node
    from .nodes.normalize_first_sentence import make_normalize_first_sentence_node
    from .nodes.assemble_payload import make_assemble_payload_node
except ImportError:  # pragma: no cover
    _current = str(Path(__file__).resolve().parent)
    if _current not in sys.path:
        sys.path.insert(0, _current)
    from epub_mcp import EbooklibEPUBMCPClient  # type: ignore
    from nodes.fetch_metadata import make_fetch_metadata_node  # type: ignore
    from nodes.fetch_table_of_contents import make_fetch_table_of_contents_node  # type: ignore
    from nodes.fetch_chapter_content_raw import make_fetch_chapter_content_raw_node  # type: ignore
    from nodes.construct_book_structure import make_construct_book_structure_node  # type: ignore
    from nodes.normalize_titles import make_normalize_titles_node  # type: ignore
    from nodes.extract_text import make_extract_text_node  # type: ignore
    from nodes.normalize_first_sentence import make_normalize_first_sentence_node  # type: ignore
    from nodes.assemble_payload import make_assemble_payload_node  # type: ignore

logger = logging.getLogger(__name__)


def _build_graph(
    mcp_client: EbooklibEPUBMCPClient,
    *,
    cache_path: Optional[Path] = None,
    debug_output_path: Optional[Path] = None,
    first_sentence_debug_path: Optional[Path] = None,
) -> Any:
    graph = StateGraph(EPUBAgentState)

    graph.add_node("fetch_metadata", make_fetch_metadata_node(mcp_client))
    graph.add_node("fetch_table_of_contents", make_fetch_table_of_contents_node(mcp_client))
    graph.add_node("fetch_chapter_content_raw", make_fetch_chapter_content_raw_node(mcp_client))
    graph.add_node("construct_book_structure", make_construct_book_structure_node(cache_path=cache_path))
    graph.add_node("normalize_titles", make_normalize_titles_node(cache_path=cache_path))
    graph.add_node("extract_text", make_extract_text_node(
        mcp_client,
        cache_path=cache_path,
        debug_output_path=debug_output_path,
    ))
    graph.add_node("normalize_first_sentence", make_normalize_first_sentence_node(
        cache_path=cache_path,
        debug_output_path=first_sentence_debug_path,
    ))
    graph.add_node("assemble_payload", make_assemble_payload_node())

    graph.add_edge("__start__", "fetch_metadata")
    graph.add_edge("fetch_metadata", "fetch_table_of_contents")
    graph.add_edge("fetch_table_of_contents", "fetch_chapter_content_raw")
    graph.add_edge("fetch_chapter_content_raw", "construct_book_structure")
    graph.add_edge("construct_book_structure", "normalize_titles")
    graph.add_edge("normalize_titles", "extract_text")
    graph.add_edge("extract_text", "normalize_first_sentence")
    graph.add_edge("normalize_first_sentence", "assemble_payload")
    graph.add_edge("assemble_payload", END)

    return graph.compile()


def _assemble_debug_payload(state: EPUBAgentState) -> Dict[str, Any]:
    toc_entries: List[Dict[str, Any]] = list(state.get("toc_entries") or [])
    chapters: List[Dict[str, Any]] = list(state.get("chapters") or [])

    chapter_list = []
    for idx, chapter in enumerate(chapters):
        toc_entry = toc_entries[idx] if idx < len(toc_entries) else {}
        chapter_list.append(
            {
                "chapter_number": chapter.get("chapter_number", idx + 1),
                "title": chapter.get("chapter_title", toc_entry.get("title", "")),
                "href": toc_entry.get("href", ""),
                "anchor": toc_entry.get("anchor", None),
                "content_text": chapter.get("content_text", ""),
            }
        )

    return {
        "metadata": state.get("metadata") or {},
        "chapters": chapter_list,
    }


class EPUBAgent:
    """Lightweight LangGraph agent for extracting raw EPUB content."""

    def __init__(self, mcp_client: Optional[EbooklibEPUBMCPClient] = None) -> None:
        self._mcp_client = mcp_client or EbooklibEPUBMCPClient()

    def run_epub_agent(
        self,
        epub_path: str,
        options: Optional[Dict[str, str]] = None,
        debug_mode: bool = True,
        debug_path: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Run the raw EPUB extraction pipeline.

        Parameters
        ----------
        epub_path:
            Path to the .epub file.
        options:
            Key/value options (reserved for future use).
        debug_mode:
            When True, write debug files to debug_path.
        debug_path:
            Directory where debug output is saved. Required when debug_mode is True.
        """
        if debug_mode and not debug_path:
            raise ValueError("debug_path is required when debug_mode is True")

        options = options or {}
        epub_path_str = str(Path(epub_path).expanduser().resolve())
        epub_stem = Path(epub_path_str).stem

        cache_path: Optional[Path] = None
        debug_output_path: Optional[Path] = None
        first_sentence_debug_path: Optional[Path] = None
        if debug_mode:
            out_dir = Path(debug_path).expanduser()
            out_dir.mkdir(parents=True, exist_ok=True)
            cache_path = out_dir / f"{epub_stem}_llm_cache.json"
            debug_output_path = out_dir / f"{epub_stem}_cleaned.json"
            first_sentence_debug_path = out_dir / f"{epub_stem}_first_sentence.json"

        graph = _build_graph(
            self._mcp_client,
            cache_path=cache_path,
            debug_output_path=debug_output_path,
            first_sentence_debug_path=first_sentence_debug_path,
        )
        initial_state: EPUBAgentState = {"epub_path": epub_path_str}

        final_state = graph.invoke(initial_state)

        # Write final result next to the epub file
        result = final_state.get("result", {})
        epub_dir = Path(epub_path_str).parent
        result_file = epub_dir / f"{epub_stem}.json"
        result_file.parent.mkdir(parents=True, exist_ok=True)
        with result_file.open("w", encoding="utf-8") as fh:
            json.dump(result, fh, indent=2, ensure_ascii=False)
        logger.info("Result written to %s", result_file)

        if debug_mode:
            payload = _assemble_debug_payload(final_state)
            out_file = Path(debug_path).expanduser() / (epub_stem + "_raw.json")
            with out_file.open("w", encoding="utf-8") as fh:
                json.dump(payload, fh, indent=2, ensure_ascii=False)
            logger.info("Debug output written to %s", out_file)

        return result


def _configure_logging(level: str = "DEBUG") -> None:
    root = logging.getLogger()
    if not root.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s [%(name)s] %(message)s"))
        root.addHandler(handler)
    root.setLevel(level.upper())


def _cli() -> int:  # pragma: no cover
    parser = argparse.ArgumentParser(
        description="Extract raw EPUB content and save epub_raw.json.",
    )
    parser.add_argument("epub_path", help="Path to the .epub file.")
    parser.add_argument(
        "--debug-path",
        required=True,
        help="Directory where epub_raw.json will be written.",
    )
    parser.add_argument(
        "--no-debug",
        action="store_true",
        help="Disable debug output (epub_raw.json will not be written).",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        help="Logging level (default: DEBUG).",
    )
    args = parser.parse_args()

    _configure_logging(args.log_level)

    agent = EPUBAgent()
    agent.run_epub_agent(
        epub_path=args.epub_path,
        debug_mode=not args.no_debug,
        debug_path=args.debug_path,
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(_cli())
