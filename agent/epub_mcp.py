from __future__ import annotations

import base64
import logging
import posixpath
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, TypedDict

import ebooklib
from ebooklib import epub

logger = logging.getLogger(__name__)

_ID_ATTRIBUTE_RE = re.compile(r'id\s*=\s*["\']([^"\']+)["\']', re.IGNORECASE)


class TableOfContentsEntry(TypedDict, total=False):
    """Single entry representing a TOC item or spine document."""

    title: str
    href: str
    anchor: Optional[str]


def _normalize_whitespace(value: str) -> str:
    """Collapse consecutive whitespace and trim the resulting string."""
    return re.sub(r"\s+", " ", value).strip()


def _extract_chapter_text(
    raw_bytes: bytes,
    anchor: Optional[str],
    next_anchor: Optional[str] = None,
) -> str:
    """Return raw HTML limited by optional anchor boundaries within the same document."""
    html_text = raw_bytes.decode("utf-8", errors="ignore")
    start_index = 0
    anchor_match: Optional[re.Match[str]] = None
    if anchor:
        anchor_pattern = re.compile(
            r'id\s*=\s*["\']' + re.escape(anchor) + r'["\']',
            re.IGNORECASE,
        )
        anchor_match = anchor_pattern.search(html_text)
        if not anchor_match:
            logger.warning("Anchor '%s' not found; using entire document.", anchor)
            return html_text

        tag_start = html_text.rfind("<", 0, anchor_match.start())
        start_index = tag_start if tag_start != -1 else anchor_match.start()

    end_index = len(html_text)
    end_match: Optional[re.Match[str]] = None
    if next_anchor:
        next_pattern = re.compile(
            r'id\s*=\s*["\']' + re.escape(next_anchor) + r'["\']',
            re.IGNORECASE,
        )
        search_from = anchor_match.end() if anchor_match else start_index
        end_match = next_pattern.search(html_text, search_from)
        if not end_match and anchor_match:
            end_match = _ID_ATTRIBUTE_RE.search(html_text, anchor_match.end())
    if end_match:
        candidate_end = html_text.rfind("<", 0, end_match.start())
        end_index = candidate_end if candidate_end != -1 else end_match.start()

    snippet = html_text[start_index:end_index]
    if snippet.strip():
        return snippet

    return html_text


def _flatten_toc_entries(entries: List[Any]) -> List[Tuple[str, str]]:
    """Flatten EbookLib TOC structures into (title, href) pairs."""
    flattened: List[Tuple[str, str]] = []
    for entry in entries:
        if isinstance(entry, epub.Link):
            title = _normalize_whitespace(entry.title or "")
            href = entry.href or ""
            if href:
                flattened.append((title, href))
        elif isinstance(entry, tuple) and entry:
            head = entry[0]
            children = entry[1] if len(entry) > 1 else []
            if isinstance(head, epub.Section):
                title = _normalize_whitespace(head.title or "")
                href = head.href or ""
                if href:
                    flattened.append((title, href))
                if isinstance(children, (list, tuple)):
                    flattened.extend(_flatten_toc_entries(list(children)))
            else:
                flattened.extend(_flatten_toc_entries([head]))
                if isinstance(children, (list, tuple)):
                    flattened.extend(_flatten_toc_entries(list(children)))
        elif isinstance(entry, epub.Section):
            title = _normalize_whitespace(entry.title or "")
            href = entry.href or ""
            if href:
                flattened.append((title, href))
        elif isinstance(entry, list):
            flattened.extend(_flatten_toc_entries(entry))
    return flattened


def _build_href_lookup(book: epub.EpubBook) -> Dict[str, epub.EpubItem]:
    lookup: Dict[str, epub.EpubItem] = {}
    for item in book.get_items():
        href = item.get_name()
        if not href:
            continue
        lookup[posixpath.normpath(href)] = item
    return lookup


def _split_href(href: str) -> Tuple[str, Optional[str]]:
    if not href:
        return "", None
    if "#" in href:
        path, fragment = href.split("#", 1)
        return path, fragment or None
    return href, None


def _normalize_doc_path(path: str) -> str:
    return posixpath.normpath(path) if path else ""


def _extract_cover_image(book: epub.EpubBook) -> Optional[Tuple[str, str]]:
    """Return cover image data encoded as base64 along with its media type."""
    candidates: List[Tuple[int, str, str, epub.EpubItem]] = []
    for item in book.get_items():
        media_type = (item.media_type or "").lower()
        if not media_type.startswith("image/"):
            continue
        href = item.get_name() or ""
        if not href:
            continue
        properties = {
            prop.casefold()
            for prop in getattr(item, "properties", [])
            if isinstance(prop, str)
        }
        priority = 2
        if item.get_type() == ebooklib.ITEM_COVER or "cover-image" in properties:
            priority = 0
        elif "cover" in (item.get_id() or "").casefold() or "cover" in href.casefold():
            priority = 1
        candidates.append((priority, href, media_type, item))

    if not candidates:
        return None

    candidates.sort(key=lambda entry: entry[0])
    for _, href, media_type, item in candidates:
        data = item.get_content()
        if not data:
            continue
        encoded = base64.b64encode(data).decode("ascii")
        return encoded, media_type
    return None


def _collect_metadata(book: epub.EpubBook) -> Dict[str, Any]:
    """Extract a subset of OPF metadata with attention to common EPUB fields."""

    def _meta(namespace: str, name: str) -> List[Tuple[str, Dict[str, str]]]:
        try:
            raw_entries = book.get_metadata(namespace, name)
        except KeyError:
            return []
        entries: List[Tuple[str, Dict[str, str]]] = []
        for value, attrs in raw_entries:
            text = _normalize_whitespace(value or "")
            if not text:
                continue
            entries.append((text, dict(attrs or {})))
        return entries

    def _attr_value(attrs: Dict[str, str], name: str) -> Optional[str]:
        """
        Return an attribute value by name, ignoring namespace prefixes such as `opf:`.
        EbookLib may expose attributes with namespaced prefixes or with varying case.
        """
        if not attrs:
            return None
        normalized = name.casefold()
        # Direct lookups first to avoid iterating in common cases.
        for candidate in (name, name.lower(), name.upper()):
            if candidate in attrs and attrs[candidate]:
                return attrs[candidate]
        for key, value in attrs.items():
            if not value or not isinstance(key, str):
                continue
            lowered = key.casefold()
            if lowered == normalized:
                return value
            if ":" in key:
                _, suffix = key.split(":", 1)
                if suffix.casefold() == normalized and value:
                    return value
            if key.startswith("{") and "}" in key:
                suffix = key.split("}", 1)[1]
                if suffix.casefold() == normalized and value:
                    return value
        return None

    metadata: Dict[str, Any] = {}

    titles = [value for value, _ in _meta("DC", "title")]
    if titles:
        metadata["title"] = titles[0]
        metadata["titles"] = titles

    creators = []
    for value, attrs in _meta("DC", "creator"):
        creators.append(
            {
                "name": value,
                "role": _attr_value(attrs, "role"),
                "file_as": _attr_value(attrs, "file-as"),
            },
        )
    if creators:
        metadata["creators"] = creators

    contributors = []
    for value, attrs in _meta("DC", "contributor"):
        contributors.append(
            {
                "name": value,
                "role": _attr_value(attrs, "role"),
            },
        )
    if contributors:
        metadata["contributors"] = contributors

    subjects = [value for value, _ in _meta("DC", "subject")]
    if subjects:
        metadata["subjects"] = subjects

    languages = [value for value, _ in _meta("DC", "language")]
    if languages:
        metadata["languages"] = languages

    publishers = [value for value, _ in _meta("DC", "publisher")]
    if publishers:
        metadata["publishers"] = publishers

    descriptions = [value for value, _ in _meta("DC", "description")]
    if descriptions:
        metadata["description"] = descriptions[0]

    rights = [value for value, _ in _meta("DC", "rights")]
    if rights:
        metadata["rights"] = rights[0]

    dates = [value for value, _ in _meta("DC", "date")]
    if dates:
        metadata["date"] = dates[0]

    identifiers = []
    for value, attrs in _meta("DC", "identifier"):
        identifiers.append(
            {
                "value": value,
                "scheme": _attr_value(attrs, "scheme"),
            },
        )
    if identifiers:
        metadata["identifiers"] = identifiers
        for entry in identifiers:
            scheme = (entry.get("scheme") or "").lower()
            if scheme == "isbn":
                metadata["isbn"] = entry["value"]
                break

    return {key: value for key, value in metadata.items() if value not in (None, [], {}, "")}


@dataclass
class _CachedBook:
    """Container for data reused across multiple MCP calls."""

    book: epub.EpubBook
    href_lookup: Dict[str, epub.EpubItem]
    spine_order: Optional[List[str]] = None
    toc_from_opf: Optional[List[TableOfContentsEntry]] = None
    toc_from_ncx: Optional[List[TableOfContentsEntry]] = None
    metadata: Optional[Dict[str, Any]] = None
    cover_image: Optional[str] = None
    cover_image_media_type: Optional[str] = None


class EbooklibEPUBMCPClient:
    """EPUB MCP client that extracts data directly with EbookLib."""

    def __init__(self) -> None:
        self._cache: Dict[str, _CachedBook] = {}

    def _cache_key(self, epub_path: str) -> str:
        return str(Path(epub_path).expanduser().resolve())

    def _resolve_cache(self, epub_path: str) -> _CachedBook:
        key = self._cache_key(epub_path)
        cached = self._cache.get(key)
        if cached:
            return cached
        book = epub.read_epub(key)
        cached = _CachedBook(book=book, href_lookup=_build_href_lookup(book))
        self._cache[key] = cached
        return cached

    def _ensure_opf_spine(self, cached: _CachedBook) -> List[str]:
        if cached.spine_order is not None:
            return cached.spine_order
        spine: List[str] = []
        for item_id, linear in cached.book.spine:
            if (linear or "yes").lower() == "no":
                continue
            if item_id:
                spine.append(item_id)
        cached.spine_order = spine
        return spine

    def _compute_toc_from_opf(
        self,
        epub_path: str,
        cached: _CachedBook,
    ) -> List[TableOfContentsEntry]:
        if cached.toc_from_opf is not None:
            return cached.toc_from_opf

        book = cached.book
        spine_ids = self._ensure_opf_spine(cached)
        flat_toc = _flatten_toc_entries(list(book.toc or []))
        toc_map: Dict[str, List[Tuple[str, Optional[str]]]] = {}
        for title, href in flat_toc:
            path, anchor = _split_href(href)
            normalized = _normalize_doc_path(path)
            toc_map.setdefault(normalized, []).append((title, anchor))

        chapters: List[TableOfContentsEntry] = []
        for item_id in spine_ids:
            manifest_item = book.get_item_with_id(item_id)
            if not manifest_item:
                logger.debug("Spine item '%s' not found in manifest.", item_id)
                continue
            doc_href = manifest_item.get_name() or ""
            normalized_doc_href = _normalize_doc_path(doc_href)
            available = toc_map.get(normalized_doc_href)
            if available:
                for title, anchor in available:
                    combined_href = doc_href
                    if anchor:
                        combined_href = f"{doc_href}#{anchor}"
                    chapters.append(
                        {
                            "title": title,
                            "href": combined_href,
                            "anchor": anchor,
                        },
                    )
            else:
                chapters.append(
                    {
                        "title": "",
                        "href": doc_href,
                        "anchor": None,
                    },
                )

        cached.toc_from_opf = chapters
        return chapters

    def _compute_toc_from_ncx(self, cached: _CachedBook) -> List[TableOfContentsEntry]:
        if cached.toc_from_ncx is not None:
            return cached.toc_from_ncx

        book = cached.book
        flat_toc = _flatten_toc_entries(list(book.toc or []))
        chapters: List[TableOfContentsEntry] = []
        for title, href in flat_toc:
            path, anchor = _split_href(href)
            normalized_path = _normalize_doc_path(path)
            combined = normalized_path
            if anchor:
                combined = f"{normalized_path}#{anchor}" if normalized_path else f"#{anchor}"
            chapters.append(
                {
                    "title": title,
                    "href": combined,
                    "anchor": anchor,
                },
            )
        cached.toc_from_ncx = chapters
        return chapters

    def get_table_of_contents(
        self,
        epub_path: str,
        source: str,
    ) -> Optional[Dict[str, Any]]:
        try:
            cached = self._resolve_cache(epub_path)
        except FileNotFoundError:
            logger.error("EPUB file not found: %s", epub_path)
            return None
        except Exception:  # pragma: no cover - defensive
            logger.exception("Failed to load EPUB for TOC extraction from %s", epub_path)
            return None

        requested = source.lower()
        chapters: List[TableOfContentsEntry]
        resolved_source = requested
        if requested == "opf":
            chapters = self._compute_toc_from_opf(epub_path, cached)
        else:
            chapters = self._compute_toc_from_ncx(cached)
            resolved_source = "ncx"
            if not chapters and requested == "ncx":
                return {"source": "ncx", "chapters": []}
            if not chapters:
                chapters = self._compute_toc_from_opf(epub_path, cached)
                resolved_source = "opf"

        return {"source": resolved_source, "chapters": chapters}

    def _find_next_anchor(
        self,
        epub_path: str,
        cached: _CachedBook,
        doc_href: str,
        current_anchor: Optional[str],
    ) -> Optional[str]:
        if cached.toc_from_opf is None and cached.toc_from_ncx is None:
            toc_ncx = self._compute_toc_from_ncx(cached)
            if not toc_ncx:
                self._compute_toc_from_opf(epub_path, cached)

        toc_entries = cached.toc_from_opf or cached.toc_from_ncx
        if not toc_entries:
            return None
        normalized_doc = _normalize_doc_path(doc_href)
        current_key = (current_anchor or "").strip()
        for idx, entry in enumerate(toc_entries):
            entry_href = entry.get("href", "")
            entry_path, entry_anchor = _split_href(entry_href)
            if _normalize_doc_path(entry_path) != normalized_doc:
                continue
            entry_key = (entry_anchor or "").strip()
            if entry_key != current_key:
                continue
            for subsequent in toc_entries[idx + 1 :]:
                next_path, next_anchor = _split_href(subsequent.get("href", ""))
                if _normalize_doc_path(next_path) != normalized_doc:
                    break
                if next_anchor:
                    return next_anchor
            break
        return None

    def get_chapter_content(
        self,
        epub_path: str,
        href: str,
    ) -> Optional[Dict[str, Any]]:
        try:
            cached = self._resolve_cache(epub_path)
        except FileNotFoundError:
            logger.error("EPUB file not found: %s", epub_path)
            return None
        except Exception:  # pragma: no cover - defensive
            logger.exception("Failed to load EPUB for chapter extraction from %s", epub_path)
            return None

        doc_path, anchor = _split_href(href)
        normalized = _normalize_doc_path(doc_path)
        item = cached.href_lookup.get(normalized)
        if not item:
            logger.warning("Chapter href '%s' not found in EPUB manifest.", href)
            return None

        next_anchor = self._find_next_anchor(epub_path, cached, doc_path, anchor)

        raw_bytes = item.get_content()
        text = _extract_chapter_text(raw_bytes, anchor, next_anchor)
        preview = (text or "")[:1000].replace("\n", " ").replace("\r", " ")
        logger.debug("Extracted chapter doc_path=%s anchor=%s preview=\"%s\"", doc_path, anchor, preview)
        return {"content_text": text}

    def get_metadata(self, epub_path: str) -> Optional[Dict[str, Any]]:
        try:
            cached = self._resolve_cache(epub_path)
        except FileNotFoundError:
            logger.error("EPUB file not found: %s", epub_path)
            return None
        except Exception:  # pragma: no cover - defensive
            logger.exception("Failed to load EPUB for metadata extraction from %s", epub_path)
            return None

        if cached.metadata is None:
            cached.metadata = _collect_metadata(cached.book)
        if cached.cover_image is None:
            cover = _extract_cover_image(cached.book)
            if cover:
                cached.cover_image, cached.cover_image_media_type = cover
            else:
                cached.cover_image = None
                cached.cover_image_media_type = None

        return {
            "metadata": cached.metadata,
            "cover_image": cached.cover_image,
            "cover_image_media_type": cached.cover_image_media_type,
        }


__all__ = ["EbooklibEPUBMCPClient", "TableOfContentsEntry"]
