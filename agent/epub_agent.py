from __future__ import annotations

import json
import logging
import sys
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, TypedDict

from dotenv import load_dotenv

try:
    from langgraph.graph import END, StateGraph
except ImportError as exc:  # pragma: no cover - import guard
    raise ImportError(
        "langgraph is required for the EPUB agent. Install it via 'pip install langgraph'.",
    ) from exc

load_dotenv()
if __package__ in (None, ""):
    current_dir = Path(__file__).resolve().parent
    current_str = str(current_dir)
    if current_str not in sys.path:
        sys.path.append(current_str)
    from epub_mcp import EbooklibEPUBMCPClient, TableOfContentsEntry
    from nodes import (
        EPUBTextProcessor,
        make_assemble_payload_node,
        make_chunk_chapter_content_node,
        make_construct_book_structure_node,
        make_fetch_chapter_content_node,
        make_fetch_metadata_node,
        make_fetch_table_of_contents_node,
        make_normalize_first_sentence_node,
        make_normalize_titles_node,
    )
else:
    from .epub_mcp import EbooklibEPUBMCPClient, TableOfContentsEntry
    from .nodes import (
        EPUBTextProcessor,
        make_assemble_payload_node,
        make_chunk_chapter_content_node,
        make_construct_book_structure_node,
        make_fetch_chapter_content_node,
        make_fetch_metadata_node,
        make_fetch_table_of_contents_node,
        make_normalize_first_sentence_node,
        make_normalize_titles_node,
    )

logger = logging.getLogger(__name__)
_DATA_BASE = Path(os.getenv("PODS_BASE", "")).expanduser()

class ChapterPayload(TypedDict, total=False):
    """Normalized structure for chapter content."""

    chapter_number: int
    chapter_title: str
    content_text: str
    chunks: List[str]


class EPUBAgentState(TypedDict, total=False):
    """State container shared across LangGraph nodes."""

    epub_path: str
    output_path: Optional[str]
    toc_entries: List[TableOfContentsEntry]
    toc_source: str
    chapters: List[ChapterPayload]
    metadata: Dict[str, Any]
    cover_image: Optional[str]
    cover_image_media_type: Optional[str]
    errors: List[str]
    result: Dict[str, Any]
    chunk_size: int
    preview_path: str
    ignore_classes: List[str]
    ai_extract_text: bool
    pods_base: str


def _build_graph(
    mcp_client: EbooklibEPUBMCPClient,
    *,
    chunk_size: int,
    ignore_classes: Optional[List[str]] = None,
    ai_extract_text: bool = False,
) -> StateGraph:
    graph = StateGraph(EPUBAgentState)
    normalized_chunk_size = max(1, chunk_size)
    cleaned_ignore_classes = list(ignore_classes or [])
    text_processor = EPUBTextProcessor(ai_extract_text=ai_extract_text)

    fetch_table_of_contents = make_fetch_table_of_contents_node(mcp_client)
    fetch_chapter_content = make_fetch_chapter_content_node(
        mcp_client,
        ignore_classes=cleaned_ignore_classes,
    )
    fetch_metadata = make_fetch_metadata_node(mcp_client)
    normalize_titles = make_normalize_titles_node(text_processor)
    normalize_first_sentence = make_normalize_first_sentence_node(text_processor)
    construct_book_structure = make_construct_book_structure_node()
    chunk_chapter_content = make_chunk_chapter_content_node(
        text_processor=text_processor,
        default_chunk_size=normalized_chunk_size,
    )
    assemble_payload = make_assemble_payload_node()
    
    graph.add_node("fetch_metadata", fetch_metadata)
    graph.add_node("fetch_table_of_contents", fetch_table_of_contents)
    graph.add_node("fetch_chapter_content", fetch_chapter_content)
    graph.add_node("construct_book_structure", construct_book_structure)
    graph.add_node("normalize_titles", normalize_titles)
    graph.add_node("normalize_first_sentence", normalize_first_sentence)
    graph.add_node("chunk_chapter_content", chunk_chapter_content)
    graph.add_node("assemble_payload", assemble_payload)

    graph.add_edge("__start__", "fetch_metadata")
    graph.add_edge("fetch_metadata", "fetch_table_of_contents")
    graph.add_edge("fetch_table_of_contents", "fetch_chapter_content")
    graph.add_edge("fetch_chapter_content", "construct_book_structure")
    graph.add_edge("construct_book_structure", "normalize_titles")
    graph.add_edge("normalize_titles", "normalize_first_sentence")
    graph.add_edge("normalize_first_sentence", "chunk_chapter_content")
    graph.add_edge("chunk_chapter_content", "assemble_payload")
    graph.add_edge("assemble_payload", END)

    return graph.compile()


class EPUBAgent:
    """LangGraph-based orchestration layer for EPUB parsing."""

    def __init__(
        self,
        mcp_client: Optional[EbooklibEPUBMCPClient] = None,
        *,
        chunk_size: int = 2000,
        ignore_classes: Optional[List[str]] = None,
        ai_extract_text: bool = False,
    ) -> None:
        self._mcp_client = mcp_client or EbooklibEPUBMCPClient()
        self._chunk_size = max(1, chunk_size)
        cleaned_ignore: List[str] = []
        seen: set[str] = set()
        if ignore_classes:
            for entry in ignore_classes:
                if not isinstance(entry, str):
                    continue
                stripped = entry.strip()
                if not stripped:
                    continue
                lowered = stripped.lower()
                if lowered in seen:
                    continue
                cleaned_ignore.append(stripped)
                seen.add(lowered)
        self._ignore_classes = cleaned_ignore
        self._ai_extract_text = bool(ai_extract_text)
        self._graph = _build_graph(
            self._mcp_client,
            chunk_size=self._chunk_size,
            ignore_classes=self._ignore_classes,
            ai_extract_text=self._ai_extract_text,
        )

    def run(
        self,
        epub_path: str | Path,
        output_path: Optional[str | Path] = None,
        *,
        preview_path: Optional[str | Path] = None,
    ) -> Dict[str, Any]:
        epub_path_str = str(Path(epub_path).expanduser().resolve())
        output_path_str = str(Path(output_path).expanduser()) if output_path else None
        preview_path_str = str(Path(preview_path).expanduser()) if preview_path else None
        initial_state: EPUBAgentState = {
            "epub_path": epub_path_str,
            "output_path": output_path_str,
            "chunk_size": self._chunk_size,
            "pods_base": str(_DATA_BASE),
        }
        if preview_path_str:
            initial_state["preview_path"] = preview_path_str
        if self._ignore_classes:
            initial_state["ignore_classes"] = list(self._ignore_classes)
        if self._ai_extract_text:
            initial_state["ai_extract_text"] = True

        final_state = self._graph.invoke(initial_state)
        result = final_state.get("result", {})
        if output_path_str:
            self._write_result(Path(output_path_str), result)
        return result

    @staticmethod
    def _write_result(path: Path, payload: Dict[str, Any]) -> None:
        path = path.expanduser()
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, ensure_ascii=False)


def run_epub_agent(
    book_title: str,
    *,
    mcp_client: Optional[EbooklibEPUBMCPClient] = None,
    chunk_size: int = 2000,
    ignore_classes: Optional[List[str]] = None,
    ai_extract_text: bool = False,
) -> Dict[str, Any]:
    """
    Convenience wrapper that instantiates and executes the EPUB agent.

    Parameters
    ----------
    book_title:
        Title/slug used to locate the EPUB under ``./data/{book_title}/book.epub``.
    mcp_client:
        Custom MCP client implementation. When omitted, a default EbookLib-backed client is used.
    chunk_size:
        Maximum character length for chapter content chunks stored alongside the original text.
    ignore_classes:
        Optional list of CSS class names whose elements should be ignored during text extraction.
    ai_extract_text:
        When True, send each generated chunk through the AI cleanup prompt for TTS readiness.

    Notes
    -----
    A preview JSON containing chapter titles and first-sentence snippets is always written to
    ``./data/{book_title}/preview.json`` before chunking.
    """
    _configure_logging()
    base_dir = _DATA_BASE / book_title
    epub_path = base_dir / "book.epub"
    resolved_output_path = base_dir / "book.json"
    preview_path = base_dir / "preview.json"

    agent = EPUBAgent(
        mcp_client=mcp_client,
        chunk_size=chunk_size,
        ignore_classes=ignore_classes,
        ai_extract_text=ai_extract_text,
    )
    return agent.run(
        epub_path=epub_path,
        output_path=resolved_output_path,
        preview_path=preview_path,
    )


def _configure_logging() -> None:
    root_logger = logging.getLogger()
    if root_logger.handlers:
        return
    log_path = Path("debug.log")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    handler = logging.FileHandler(log_path, mode="a", encoding="utf-8")
    formatter = logging.Formatter("%(asctime)s %(levelname)s [%(name)s] %(message)s")
    handler.setFormatter(formatter)
    root_logger.addHandler(handler)
    root_logger.setLevel(logging.DEBUG)


def _parse_ignore_classes(raw_values: Optional[List[str]]) -> Optional[List[str]]:
    if not raw_values:
        return None
    cleaned: List[str] = []
    seen: set[str] = set()
    for value in raw_values:
        if not isinstance(value, str):
            continue
        parts = [part.strip() for part in value.split(",")]
        for part in parts:
            if not part:
                continue
            lowered = part.lower()
            if lowered in seen:
                continue
            cleaned.append(part)
            seen.add(lowered)
    return cleaned or None


def _create_arg_parser() -> "argparse.ArgumentParser":  # pragma: no cover - CLI helper
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate a JSON representation of an EPUB using the LangGraph agent.",
    )
    parser.add_argument(
        "book_title",
        help="Book title used to locate ./data/{book_title}/book.epub.",
    )
    parser.add_argument(
        "--log-level",
        default="DEBUG",
        help="Logging level for the agent execution (default: DEBUG).",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=200,
        help="Maximum characters per chapter chunk stored alongside the original text (default: 2000).",
    )
    parser.add_argument(
        "--ignore-class",
        action="append",
        help=(
            "Comma-separated CSS classes to ignore when extracting text. "
            "Repeat the flag to provide additional entries."
        ),
    )
    parser.add_argument(
        "--ai-extract-text",
        action="store_true",
        help=(
            "Use the AI OCR-cleanup prompt to normalize each chunk for TTS output "
            "(slower, requires OpenRouter credentials)."
        ),
    )
    return parser


def _cli() -> int:  # pragma: no cover - CLI helper
    parser = _create_arg_parser()
    args = parser.parse_args()

    _configure_logging()
    logging.getLogger().setLevel(args.log_level.upper())

    result = run_epub_agent(
        args.book_title,
        chunk_size=args.chunk_size,
        ignore_classes=_parse_ignore_classes(args.ignore_class),
        ai_extract_text=args.ai_extract_text,
    )
    logger.info(
        "EPUB agent completed for '%s'; output written to %s",
        args.book_title,
        _DATA_BASE / args.book_title / "book.json",
    )
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entrypoint
    raise SystemExit(_cli())
