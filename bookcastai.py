from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent


def _run(cmd: list[str], description: str) -> None:
    """Run a subprocess command, exiting early if it fails."""
    print(f"-> {description}")
    result = subprocess.run(cmd)
    if result.returncode != 0:
        raise SystemExit(result.returncode)


def _build_epub_agent_cmd(args: argparse.Namespace) -> list[str]:
    """Construct the command for epub_agent with the supplied flags."""
    cmd = [
        sys.executable,
        str(PROJECT_ROOT / "agent" / "epub_agent.py"),
        args.book_title,
        "--log-level",
        args.log_level,
        "--chunk-size",
        str(args.chunk_size),
    ]
    for ignore in args.ignore_class or []:
        cmd.extend(["--ignore-class", ignore])
    if args.ai_extract_text:
        cmd.append("--ai-extract-text")
    return cmd


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the full BookcastAI pipeline: EPUB parsing -> audio -> feed.",
    )
    parser.add_argument(
        "book_title",
        help="Book title used to locate ./data/{book_title}/ assets.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        help="Logging level for the EPUB agent (default: INFO).",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=200,
        help="Maximum characters per chapter chunk stored alongside the original text (default: 200).",
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
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)

    epub_agent_cmd = _build_epub_agent_cmd(args)
    epub_to_pod_cmd = [
        sys.executable,
        str(PROJECT_ROOT / "epub_to_pod.py"),
        args.book_title,
    ]
    feed_cmd = [
        sys.executable,
        str(PROJECT_ROOT / "feed.py"),
        args.book_title,
    ]

    _run(epub_agent_cmd, "Running EPUB agent")
    _run(epub_to_pod_cmd, "Converting EPUB to audio")
    _run(feed_cmd, "Generating and uploading feed")

    print("Pipeline completed successfully.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
