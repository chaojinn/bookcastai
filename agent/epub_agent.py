from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, TypedDict

try:
    from langgraph.graph import END, StateGraph
except ImportError as exc:  # pragma: no cover - import guard
    raise ImportError(
        "langgraph is required for the EPUB agent. Install it via 'pip install langgraph'.",
    ) from exc

if __package__ in (None, ""):
    current_dir = Path(__file__).resolve().parent
    current_str = str(current_dir)
    if current_str not in sys.path:
        sys.path.append(current_str)
    from epub_mcp import EbooklibEPUBMCPClient, TableOfContentsEntry
    from text_converter import EPUBTextConverter
else:
    from .epub_mcp import EbooklibEPUBMCPClient, TableOfContentsEntry
    from .text_converter import EPUBTextConverter

logger = logging.getLogger(__name__)

_SENTENCE_TERMINATORS = ".?!"
_COMMON_ABBREVIATIONS = {
    "dr",
    "mr",
    "mrs",
    "ms",
    "prof",
    "sir",
    "jr",
    "rev",
}


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
    start_chapter_number: int
    end_chapter_number: int
    selected_chapter_numbers: List[int]
    ignore_classes: List[str]


def _split_text_into_chunks(text: str, chunk_size: int) -> List[str]:
    """Split text into fixed-size chunks while preserving order."""
    size = max(1, chunk_size)
    return [text[index : index + size] for index in range(0, len(text), size)]


def _looks_like_abbreviation(text: str, punct_index: int) -> bool:
    """Return True when a trailing period is part of a known abbreviation."""
    if punct_index <= 0 or punct_index > len(text):
        return False
    word_end = punct_index
    word_start = word_end
    while word_start > 0 and text[word_start - 1].isalpha():
        word_start -= 1
    if word_start == word_end:
        return False
    candidate = text[word_start:word_end].lower()
    return candidate in _COMMON_ABBREVIATIONS


def _segment_text_by_punctuation(text: str) -> List[str]:
    """Split text whenever a punctuation terminator is encountered."""
    segments: List[str] = []
    start = 0
    idx = 0
    length = len(text)
    while idx < length:
        char = text[idx]
        if char in _SENTENCE_TERMINATORS:
            if char == "." and _looks_like_abbreviation(text, idx):
                idx += 1
                continue
            idx += 1
            while idx < length and text[idx].isspace():
                idx += 1
            segments.append(text[start:idx])
            start = idx
            continue
        idx += 1
    if start < length:
        segments.append(text[start:])
    return [segment for segment in segments if segment.strip()]


def _split_text_by_punctuation(text: str, chunk_size: int) -> List[str]:
    """Chunk text by punctuation while keeping each chunk under chunk_size."""
    size = max(1, chunk_size)
    segments = _segment_text_by_punctuation(text)
    if not segments:
        return _split_text_into_chunks(text, size)

    chunks: List[str] = []
    current_parts: List[str] = []
    current_len = 0

    def flush_current() -> None:
        nonlocal current_parts, current_len
        if not current_parts:
            return
        chunk = "".join(current_parts).strip()
        if chunk:
            chunks.append(chunk)
        current_parts = []
        current_len = 0

    for segment in segments:
        if len(segment) > size:
            flush_current()
            for piece in _split_text_into_chunks(segment, size):
                cleaned = piece.strip()
                if cleaned:
                    chunks.append(cleaned)
            continue

        if current_len + len(segment) <= size:
            current_parts.append(segment)
            current_len += len(segment)
            continue

        flush_current()
        current_parts.append(segment)
        current_len = len(segment)

    flush_current()
    return chunks


def _build_graph(
    mcp_client: EbooklibEPUBMCPClient,
    *,
    chunk_size: int,
    ignore_classes: Optional[List[str]] = None,
    preview: bool = False,
) -> StateGraph:
    graph = StateGraph(EPUBAgentState)
    text_converter = EPUBTextConverter(ignore_classes=ignore_classes or [])
    normalized_chunk_size = max(1, chunk_size)

    def fetch_table_of_contents(state: EPUBAgentState) -> EPUBAgentState:
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

        errors.append(
            "Unable to extract table of contents from either OPF or NCX manifest.",
        )
        return {"toc_entries": [], "toc_source": "", "errors": errors}

    def fetch_chapter_content(state: EPUBAgentState) -> EPUBAgentState:
        toc_entries = state.get("toc_entries", [])
        epub_path = state["epub_path"]
        errors: List[str] = list(state.get("errors", []))
        if not toc_entries:
            errors.append("Chapter extraction skipped because TOC entries are missing.")
            return {"chapters": [], "errors": errors}

        chapters: List[ChapterPayload] = []
        for number, entry in enumerate(toc_entries, start=1):
            href = entry.get("href", "")
            if not href:
                errors.append(f"Skipping chapter {number}: missing href in TOC entry.")
                continue
            response = mcp_client.get_chapter_content(epub_path, href=href)
            if not response:
                errors.append(
                    f"MCP client returned no content for chapter href '{href}'.",
                )
                continue

            content_text = response.get("content_text", "")
            chapters.append(
                {
                    "chapter_number": number,
                    "chapter_title": entry.get("title", ""),
                    "content_text": content_text,
                },
            )
        return {"chapters": chapters, "errors": errors}

    def enforce_chapter_limits_for_preview(state: EPUBAgentState) -> EPUBAgentState:
        if not preview:
            return {}
        chapters = state.get("chapters")
        if not isinstance(chapters, list) or not chapters:
            return {}
        filtered = text_converter.filter_chapters(chapters, state)
        if filtered == chapters:
            return {}
        return {"chapters": filtered}

    def fetch_metadata(state: EPUBAgentState) -> EPUBAgentState:
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

        metadata = response.get("metadata") or {}
        cover_image = response.get("cover_image")
        cover_type = response.get("cover_image_media_type")
        return {
            "metadata": metadata,
            "cover_image": cover_image,
            "cover_image_media_type": cover_type,
            "errors": errors,
        }

    def convert_text(state: EPUBAgentState) -> EPUBAgentState:
        """Clean text, enforce chapter limits, and normalize titles via OpenRoute."""
        return text_converter.convert(state)

    def chunk_chapter_content(state: EPUBAgentState) -> EPUBAgentState:
        chapters = state.get("chapters", [])
        if not chapters:
            return {}

        state_chunk_size = state.get("chunk_size")
        effective_chunk_size = (
            max(1, int(state_chunk_size)) if isinstance(state_chunk_size, int) else normalized_chunk_size
        )

        updated_chapters: List[ChapterPayload] = []
        modified = False
        for chapter in chapters:
            if not isinstance(chapter, dict):
                updated_chapters.append(chapter)
                continue
            content = chapter.get("content_text")
            if not isinstance(content, str) or not content:
                updated_chapters.append(chapter)
                continue
            chunks = _split_text_by_punctuation(content, effective_chunk_size)
            existing_chunks = chapter.get("chunks")
            if isinstance(existing_chunks, list) and existing_chunks == chunks:
                updated_chapters.append(chapter)
                continue
            chapter_copy: ChapterPayload = dict(chapter)
            chapter_copy["chunks"] = chunks
            updated_chapters.append(chapter_copy)
            modified = True
        return {"chapters": updated_chapters} if modified else {}

    def assemble_payload(state: EPUBAgentState) -> EPUBAgentState:
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

        # Retain visibility into the chapter source when available.
        toc_source = state.get("toc_source")
        if toc_source:
            payload["toc_source"] = toc_source

        errors = state.get("errors", [])
        if errors:
            payload["_warnings"] = errors
        return {"result": payload}

    graph.add_node("fetch_table_of_contents", fetch_table_of_contents)
    graph.add_node("fetch_chapter_content", fetch_chapter_content)
    graph.add_node("fetch_metadata", fetch_metadata)
    graph.add_node("enforce_chapter_limits_for_preview", enforce_chapter_limits_for_preview)
    if not preview:
        graph.add_node("convert_text", convert_text)
    graph.add_node("chunk_chapter_content", chunk_chapter_content)
    graph.add_node("assemble_payload", assemble_payload)

    graph.add_edge("__start__", "fetch_table_of_contents")
    graph.add_edge("fetch_table_of_contents", "fetch_chapter_content")
    if preview:
        graph.add_edge("fetch_chapter_content", "enforce_chapter_limits_for_preview")
        graph.add_edge("enforce_chapter_limits_for_preview", "fetch_metadata")
    else:
        graph.add_edge("fetch_chapter_content", "fetch_metadata")
    if preview:
        graph.add_edge("fetch_metadata", "chunk_chapter_content")
    else:
        graph.add_edge("fetch_metadata", "convert_text")
        graph.add_edge("convert_text", "chunk_chapter_content")
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
        start_chapter_number: int = 1,
        end_chapter_number: Optional[int] = None,
        selected_chapters: Optional[Sequence[int]] = None,
        ignore_classes: Optional[List[str]] = None,
        preview: bool = False,
    ) -> None:
        self._mcp_client = mcp_client or EbooklibEPUBMCPClient()
        self._chunk_size = max(1, chunk_size)
        self._start_chapter_number = max(1, start_chapter_number)
        self._end_chapter_number = (
            max(self._start_chapter_number, end_chapter_number)
            if isinstance(end_chapter_number, int)
            and end_chapter_number >= self._start_chapter_number
            else None
        )
        cleaned_selection: List[int] = []
        if selected_chapters:
            seen_numbers: set[int] = set()
            for value in selected_chapters:
                if not isinstance(value, int):
                    continue
                if value <= 0 or value in seen_numbers:
                    continue
                seen_numbers.add(value)
                cleaned_selection.append(value)
        cleaned_selection.sort()
        self._selected_chapters = cleaned_selection
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
        self._preview = bool(preview)
        self._graph = _build_graph(
            self._mcp_client,
            chunk_size=self._chunk_size,
            ignore_classes=self._ignore_classes,
            preview=self._preview,
        )

    def run(
        self,
        epub_path: str | Path,
        output_path: Optional[str | Path] = None,
    ) -> Dict[str, Any]:
        epub_path_str = str(Path(epub_path).expanduser().resolve())
        output_path_str = str(Path(output_path).expanduser()) if output_path else None
        initial_state: EPUBAgentState = {
            "epub_path": epub_path_str,
            "output_path": output_path_str,
            "chunk_size": self._chunk_size,
            "start_chapter_number": self._start_chapter_number,
        }
        if self._end_chapter_number is not None:
            initial_state["end_chapter_number"] = self._end_chapter_number
        if self._selected_chapters:
            initial_state["selected_chapter_numbers"] = list(self._selected_chapters)
        if self._ignore_classes:
            initial_state["ignore_classes"] = list(self._ignore_classes)

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
    epub_path: str | Path,
    output_path: Optional[str | Path] = None,
    *,
    mcp_client: Optional[EbooklibEPUBMCPClient] = None,
    chunk_size: int = 2000,
    start_chapter_number: int = 1,
    end_chapter_number: Optional[int] = None,
    selected_chapters: Optional[Sequence[int]] = None,
    ignore_classes: Optional[List[str]] = None,
    preview: bool = False,
) -> Dict[str, Any]:
    """
    Convenience wrapper that instantiates and executes the EPUB agent.

    Parameters
    ----------
    epub_path:
        Path to the source EPUB file.
    output_path:
        Optional destination path for the resulting JSON payload.
    mcp_client:
        Custom MCP client implementation. When omitted, a default EbookLib-backed client is used.
    chunk_size:
        Maximum character length for chapter content chunks stored alongside the original text.
    start_chapter_number:
        Original chapter number at which to start including chapters in the final payload (default: 1).
    end_chapter_number:
        Optional original chapter number at which to stop including chapters in the final payload.
    selected_chapters:
        Optional explicit list of original chapter numbers that must be retained (comma-separated via CLI).
    ignore_classes:
        Optional list of CSS class names whose elements should be ignored during text extraction.
    preview:
        When True, skip the conversion step for faster previews (default: False).
    """
    agent = EPUBAgent(
        mcp_client=mcp_client,
        chunk_size=chunk_size,
        start_chapter_number=start_chapter_number,
        end_chapter_number=end_chapter_number,
        selected_chapters=selected_chapters,
        ignore_classes=ignore_classes,
        preview=preview,
    )
    return agent.run(epub_path=epub_path, output_path=output_path)


def _configure_logging() -> None:
    root_logger = logging.getLogger()
    if root_logger.handlers:
        return
    handler = logging.StreamHandler(stream=sys.stdout)
    formatter = logging.Formatter("%(asctime)s %(levelname)s [%(name)s] %(message)s")
    handler.setFormatter(formatter)
    root_logger.addHandler(handler)
    root_logger.setLevel(logging.INFO)


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


def _parse_chapter_numbers(value: Optional[str]) -> Optional[List[int]]:
    if not value:
        return None
    cleaned: List[int] = []
    seen: set[int] = set()
    for raw_entry in value.split(","):
        entry = raw_entry.strip()
        if not entry:
            continue
        try:
            number = int(entry)
        except ValueError as exc:
            raise ValueError(f"Invalid chapter number '{entry}' provided to --chapters.") from exc
        if number <= 0:
            raise ValueError("Chapter numbers supplied via --chapters must be positive integers.")
        if number in seen:
            continue
        seen.add(number)
        cleaned.append(number)
    return cleaned or None


def _create_arg_parser() -> "argparse.ArgumentParser":  # pragma: no cover - CLI helper
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate a JSON representation of an EPUB using the LangGraph agent.",
    )
    parser.add_argument("epub", help="Path to the EPUB file.")
    parser.add_argument(
        "-o",
        "--output",
        help="Destination path for the generated JSON. Printed to stdout when omitted.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        help="Logging level for the agent execution (default: INFO).",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=2000,
        help="Maximum characters per chapter chunk stored alongside the original text (default: 2000).",
    )
    parser.add_argument(
        "--start-chapter-number",
        type=int,
        default=1,
        help="Original chapter number at which to begin including chapters in the output (default: 1).",
    )
    parser.add_argument(
        "--end-chapter-number",
        type=int,
        help="Optional original chapter number at which to stop including chapters in the output.",
    )
    parser.add_argument(
        "--chapters",
        help=(
            "Comma-separated list of original chapter numbers to include in the final output "
            "(e.g., 1,3,5)."
        ),
    )
    parser.add_argument(
        "--preview",
        action="store_true",
        help="Skip chapter text conversion for faster previews.",
    )
    parser.add_argument(
        "--ignore-class",
        action="append",
        help=(
            "Comma-separated CSS classes to ignore when extracting text. "
            "Repeat the flag to provide additional entries."
        ),
    )
    return parser


def _cli() -> int:  # pragma: no cover - CLI helper
    parser = _create_arg_parser()
    args = parser.parse_args()

    _configure_logging()
    logging.getLogger().setLevel(args.log_level.upper())

    try:
        chapter_numbers = _parse_chapter_numbers(args.chapters)
    except ValueError as exc:
        parser.error(str(exc))

    result = run_epub_agent(
        args.epub,
        args.output,
        chunk_size=args.chunk_size,
        start_chapter_number=args.start_chapter_number,
        end_chapter_number=args.end_chapter_number,
        selected_chapters=chapter_numbers,
        ignore_classes=_parse_ignore_classes(args.ignore_class),
        preview=args.preview,
    )
    if not args.output:
        print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entrypoint
    raise SystemExit(_cli())
