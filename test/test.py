from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Optional

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent.nodes.roman import convert_roman_numerals


def _load_book(book_title: str, base_dir: Path) -> Dict[str, Any]:
    book_path = base_dir / "data" / book_title / "book.json"
    if not book_path.exists():
        raise FileNotFoundError(f"Book file not found: {book_path}")
    with book_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _get_chapter_content(book: Dict[str, Any], chapter_number: int) -> str:
    chapters = book.get("chapters")
    if not isinstance(chapters, list):
        raise ValueError("Book JSON does not contain a valid 'chapters' list.")
    for chapter in chapters:
        if not isinstance(chapter, dict):
            continue
        number: Optional[int] = chapter.get("chapter_number")
        if isinstance(number, str) and number.isdigit():
            number = int(number)
        if number == chapter_number:
            content = chapter.get("content_text")
            if not isinstance(content, str):
                raise ValueError(f"Chapter {chapter_number} is missing 'content_text'.")
            return content
    raise ValueError(f"Chapter {chapter_number} not found.")


def _write_debug_log(root: Path, payload: Dict[str, Any]) -> None:
    log_path = root / "debug.log"
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False, indent=2) + "\n\n")


def main() -> int:
    parser = argparse.ArgumentParser(description="Convert Roman numerals in a chapter.")
    parser.add_argument("book_title", help="Book folder name under ./data/")
    parser.add_argument(
        "chapter_number",
        type=int,
        nargs="?",
        help="Chapter number to process (omit to process all chapters)",
    )
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    try:
        book = _load_book(args.book_title, root)
        if args.chapter_number is not None:
            content = _get_chapter_content(book, args.chapter_number)
            result = convert_roman_numerals(content)
            payload = {
                "book_title": args.book_title,
                "chapter_number": args.chapter_number,
                "result": result,
            }
        else:
            chapters = book.get("chapters")
            if not isinstance(chapters, list):
                raise ValueError("Book JSON does not contain a valid 'chapters' list.")
            combined: list[str] = []
            for chapter in chapters:
                if not isinstance(chapter, dict):
                    continue
                content = chapter.get("content_text")
                if isinstance(content, str):
                    combined.append(content)
            if not combined:
                raise ValueError("No chapter content available to process.")
            content_text = "\n".join(combined)
            result = convert_roman_numerals(content_text)
            payload = {
                "book_title": args.book_title,
                "chapter_number": None,
                "result": result,
            }
        _write_debug_log(root, payload)
        print("Conversion complete. Result written to debug.log")
        return 0
    except Exception as exc:  # pragma: no cover - utility script
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
