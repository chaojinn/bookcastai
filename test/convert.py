from __future__ import annotations

import base64
import json
from pathlib import Path


def _decode_cover(folder: Path) -> None:
    book_path = folder / "book.json"
    if not book_path.exists():
        return

    try:
        data = json.loads(book_path.read_text(encoding="utf-8"))
    except Exception as exc:  # pragma: no cover - defensive parsing
        print(f"[skip] Failed to read {book_path}: {exc}")
        return

    cover_b64 = data.get("cover_image")
    media_type = (data.get("cover_image_media_type") or "").lower()
    if not cover_b64 or not media_type:
        return

    ext = None
    if "png" in media_type:
        ext = "png"
    elif "jpeg" in media_type or "jpg" in media_type:
        ext = "jpg"
    if ext is None:
        print(f"[skip] Unsupported media type {media_type} in {book_path}")
        return

    try:
        decoded = base64.b64decode(cover_b64)
    except Exception as exc:  # pragma: no cover
        print(f"[skip] Failed to decode cover in {book_path}: {exc}")
        return

    out_path = folder / f"cover.{ext}"
    out_path.write_bytes(decoded)
    print(f"[ok] Wrote {out_path}")


def main() -> int:
    base_dir = Path(__file__).resolve().parent.parent / "data"
    if not base_dir.exists():
        print(f"[skip] Data directory not found: {base_dir}")
        return 0

    for subdir in sorted(base_dir.iterdir()):
        if not subdir.is_dir():
            continue
        _decode_cover(subdir)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
