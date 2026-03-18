from __future__ import annotations

import argparse
import json
import logging
import sys
import uuid
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, TypedDict

try:
    from langgraph.graph import END, StateGraph
    from langgraph.checkpoint.memory import MemorySaver
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
    # extract_text split
    cleanup_rules: List[str]
    raw_html_map: Dict[str, str]
    approved_rules: List[str]
    rule_previews: List[Dict[str, Any]]


try:  # pragma: no cover - import shim for script execution
    from .epub_mcp import EbooklibEPUBMCPClient
    from .nodes.fetch_metadata import make_fetch_metadata_node
    from .nodes.fetch_table_of_contents import make_fetch_table_of_contents_node
    from .nodes.fetch_chapter_content_raw import make_fetch_chapter_content_raw_node
    from .nodes.construct_book_structure import make_construct_book_structure_node
    from .nodes.normalize_titles import make_normalize_titles_node
    from .nodes.generate_rules import make_generate_rules_node
    from .nodes.review_rules import make_review_rules_node
    from .nodes.apply_rules import make_apply_rules_node
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
    from nodes.generate_rules import make_generate_rules_node  # type: ignore
    from nodes.review_rules import make_review_rules_node  # type: ignore
    from nodes.apply_rules import make_apply_rules_node  # type: ignore
    from nodes.normalize_first_sentence import make_normalize_first_sentence_node  # type: ignore
    from nodes.assemble_payload import make_assemble_payload_node  # type: ignore

logger = logging.getLogger(__name__)


def _build_graph(
    mcp_client: EbooklibEPUBMCPClient,
    *,
    cache_path: Optional[Path] = None,
    debug_output_path: Optional[Path] = None,
    first_sentence_debug_path: Optional[Path] = None,
    publish_progress: Optional[Callable[[int, str], None]] = None,
) -> Any:
    node_sequence = [
        ("fetch_metadata", make_fetch_metadata_node(mcp_client), "Fetched metadata"),
        ("fetch_table_of_contents", make_fetch_table_of_contents_node(mcp_client), "Fetched table of contents"),
        ("fetch_chapter_content_raw", make_fetch_chapter_content_raw_node(mcp_client), "Fetched chapter content"),
        ("construct_book_structure", make_construct_book_structure_node(cache_path=cache_path), "Constructed book structure"),
        ("normalize_titles", make_normalize_titles_node(cache_path=cache_path), "Normalized titles"),
        ("generate_rules", make_generate_rules_node(mcp_client, cache_path=cache_path), "Generated cleanup rules"),
        ("review_rules", make_review_rules_node(), "Reviewed rules"),
        ("apply_rules", make_apply_rules_node(debug_output_path=debug_output_path), "Applied cleanup rules"),
        ("normalize_first_sentence", make_normalize_first_sentence_node(cache_path=cache_path, debug_output_path=first_sentence_debug_path), "Normalized first sentences"),
        ("assemble_payload", make_assemble_payload_node(), "Assembled payload"),
    ]
    total_nodes = len(node_sequence)

    def _wrap_node(
        name: str,
        node_fn: Callable[[EPUBAgentState], EPUBAgentState],
        index: int,
        message: str,
    ) -> Callable[[EPUBAgentState], EPUBAgentState]:
        def _wrapped(state: EPUBAgentState) -> EPUBAgentState:
            result = node_fn(state)
            if publish_progress is not None:
                try:
                    progress = int(round(((index + 1) / total_nodes) * 100))
                    publish_progress(progress, message)
                except Exception:
                    logger.exception("Progress callback failed for node %s", name)
            return result
        return _wrapped

    graph = StateGraph(EPUBAgentState)
    for idx, (node_name, node_fn, message) in enumerate(node_sequence):
        graph.add_node(node_name, _wrap_node(node_name, node_fn, idx, message))

    graph.add_edge("__start__", "fetch_metadata")
    graph.add_edge("fetch_metadata", "fetch_table_of_contents")
    graph.add_edge("fetch_table_of_contents", "fetch_chapter_content_raw")
    graph.add_edge("fetch_chapter_content_raw", "construct_book_structure")
    graph.add_edge("construct_book_structure", "normalize_titles")
    graph.add_edge("normalize_titles", "generate_rules")
    graph.add_edge("generate_rules", "review_rules")
    graph.add_edge("review_rules", "apply_rules")
    graph.add_edge("apply_rules", "normalize_first_sentence")
    graph.add_edge("normalize_first_sentence", "assemble_payload")
    graph.add_edge("assemble_payload", END)

    return graph.compile(
        checkpointer=MemorySaver(),
        interrupt_before=["review_rules"],
    )


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


def _write_result(final_state: EPUBAgentState, epub_path_str: str, debug_mode: bool, debug_path: Optional[str]) -> Dict[str, Any]:
    """Write result JSON next to epub and (optionally) debug output. Returns result dict."""
    result = final_state.get("result", {})
    epub_dir = Path(epub_path_str).parent
    epub_stem = Path(epub_path_str).stem
    result_file = epub_dir / f"{epub_stem}.json"
    result_file.parent.mkdir(parents=True, exist_ok=True)
    with result_file.open("w", encoding="utf-8") as fh:
        json.dump(result, fh, indent=2, ensure_ascii=False)
    logger.info("Result written to %s", result_file)

    if debug_mode and debug_path:
        payload = _assemble_debug_payload(final_state)
        out_file = Path(debug_path).expanduser() / (epub_stem + "_raw.json")
        with out_file.open("w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2, ensure_ascii=False)
        logger.info("Debug output written to %s", out_file)

    return result


class EPUBAgent:
    """LangGraph agent for extracting raw EPUB content with optional HITL rule review."""

    def __init__(self, mcp_client: Optional[EbooklibEPUBMCPClient] = None) -> None:
        self._mcp_client = mcp_client or EbooklibEPUBMCPClient()
        self._compiled_graph: Optional[Any] = None
        self._thread_config: Optional[Dict[str, Any]] = None
        self._epub_path_str: Optional[str] = None
        self._debug_mode: bool = False
        self._debug_path: Optional[str] = None

    def run_phase1(
        self,
        epub_path: str,
        debug_mode: bool = True,
        debug_path: Optional[str] = None,
        publish_progress: Optional[Callable[[int, str], None]] = None,
    ) -> None:
        """Run the pipeline up to the HITL interrupt (before review_rules).

        After this call, ``get_review_data()`` can be used to retrieve rule previews,
        and ``run_phase2()`` resumes execution after human approval.
        """
        if debug_mode and not debug_path:
            raise ValueError("debug_path is required when debug_mode is True")

        self._epub_path_str = str(Path(epub_path).expanduser().resolve())
        self._debug_mode = debug_mode
        self._debug_path = debug_path
        epub_stem = Path(self._epub_path_str).stem

        cache_path: Optional[Path] = None
        debug_output_path: Optional[Path] = None
        first_sentence_debug_path: Optional[Path] = None
        if debug_mode and debug_path:
            out_dir = Path(debug_path).expanduser()
            out_dir.mkdir(parents=True, exist_ok=True)
            cache_path = out_dir / f"{epub_stem}_llm_cache.json"
            debug_output_path = out_dir / f"{epub_stem}_cleaned.json"
            first_sentence_debug_path = out_dir / f"{epub_stem}_first_sentence.json"

        self._compiled_graph = _build_graph(
            self._mcp_client,
            cache_path=cache_path,
            debug_output_path=debug_output_path,
            first_sentence_debug_path=first_sentence_debug_path,
            publish_progress=publish_progress,
        )

        thread_id = str(uuid.uuid4())
        self._thread_config = {"configurable": {"thread_id": thread_id}}

        initial_state: EPUBAgentState = {"epub_path": self._epub_path_str}
        self._compiled_graph.invoke(initial_state, config=self._thread_config)
        # Returns here because of interrupt_before=["review_rules"]

    def get_review_data(self) -> List[Dict[str, Any]]:
        """Return rule_previews from the checkpointed state (call after run_phase1)."""
        if self._compiled_graph is None or self._thread_config is None:
            return []
        state_snapshot = self._compiled_graph.get_state(self._thread_config)
        return list(state_snapshot.values.get("rule_previews") or [])

    def run_phase2(self, approved_rules: List[str]) -> Dict[str, Any]:
        """Inject approved rules and resume graph execution to completion."""
        if self._compiled_graph is None or self._thread_config is None:
            raise RuntimeError("run_phase1 must be called before run_phase2")

        self._compiled_graph.update_state(
            self._thread_config,
            {"approved_rules": approved_rules},
        )
        final_state = self._compiled_graph.invoke(None, config=self._thread_config)

        return _write_result(
            final_state,
            self._epub_path_str,
            self._debug_mode,
            self._debug_path,
        )

    def run_epub_agent(
        self,
        epub_path: str,
        options: Optional[Dict[str, str]] = None,  # reserved for future use
        debug_mode: bool = True,
        debug_path: Optional[str] = None,
        publish_progress: Optional[Callable[[int, str], None]] = None,
    ) -> Dict[str, Any]:
        """
        Run the full EPUB extraction pipeline (non-interactive, auto-approves all rules).

        This preserves the original CLI / job-queue interface. For interactive HITL,
        use ``run_phase1`` / ``get_review_data`` / ``run_phase2`` directly.
        """
        self.run_phase1(
            epub_path,
            debug_mode=debug_mode,
            debug_path=debug_path,
            publish_progress=publish_progress,
        )
        review_data = self.get_review_data()
        all_rules = [item["rule"] for item in review_data if isinstance(item.get("rule"), str)]
        return self.run_phase2(all_rules)


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
