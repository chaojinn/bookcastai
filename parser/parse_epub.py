import argparse
import base64
import html
import json
import logging
import posixpath
import re
import sys
from io import BytesIO
from pathlib import Path
from xml.etree import ElementTree as ET

import ebooklib
from ebooklib import epub

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parent.parent))
    from parser.utils.roman import replace_numeric_titles
else:
    from .utils.roman import replace_numeric_titles


class EPUBParsingError(RuntimeError):
    """Raised when an expected EPUB structure is missing or malformed."""


DEFAULT_LOG_PATH = Path("debug.log")
logger = logging.getLogger(__name__)
if not logger.handlers:
    handler = logging.FileHandler(DEFAULT_LOG_PATH, encoding="utf-8")
    formatter = logging.Formatter(
        "%(asctime)s %(levelname)s [%(name)s] %(message)s",
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.DEBUG)
    logger.propagate = False


def _normalize_whitespace(value: str) -> str:
    """Collapse consecutive whitespace and trim the resulting string."""
    return re.sub(r"\s+", " ", value).strip()


def _dedupe_leading_title(title: str, content: str) -> str:
    """Remove duplicate occurrences of the chapter title at the start of content."""
    if not title or not content:
        return content

    normalized_title = _normalize_whitespace(title)
    if not normalized_title:
        return content

    remainder = content.lstrip()
    lowered_title = normalized_title.casefold()
    removed = 0

    while remainder.casefold().startswith(lowered_title):
        after = remainder[len(normalized_title):]
        if after and not after[:1].isspace():
            break
        remainder = after.lstrip()
        removed += 1

    if removed:
        if remainder:
            return f"{normalized_title} {remainder}".strip()
        return normalized_title

    return content


_ABBREVIATION_RE = re.compile(
    r"\b(?:Mr|Ms|Mrs|Dr|Prof|Sir|Jr|Rev)\.$",
    re.IGNORECASE,
)
_SENTENCE_BOUNDARY_RE = re.compile(r"(?<=[.!?])\s+")
_LEADING_CHAPTER_MARKER_RE = re.compile(
    r"^(?P<space>\s*)(?:(?P<label>chapter)\s+)?(?P<num>(?:[MDCLXVI]+|\d+))\b",
    re.IGNORECASE,
)
_ID_ATTRIBUTE_RE = re.compile(r'id\s*=\s*["\']([^"\']+)["\']', re.IGNORECASE)
_HTML_TAG_RE = re.compile(r"<[^>]+>")

OPF_NAMESPACE = "http://www.idpf.org/2007/opf"

_NUMBER_WORDS = {
    "zero",
    "one",
    "two",
    "three",
    "four",
    "five",
    "six",
    "seven",
    "eight",
    "nine",
    "ten",
    "eleven",
    "twelve",
    "thirteen",
    "fourteen",
    "fifteen",
    "sixteen",
    "seventeen",
    "eighteen",
    "nineteen",
    "twenty",
    "thirty",
    "forty",
    "fifty",
    "sixty",
    "seventy",
    "eighty",
    "ninety",
}
_NUMBER_SCALE_WORDS = {"hundred", "thousand", "million"}
_NUMBER_ALLOWED_TOKENS = _NUMBER_WORDS | _NUMBER_SCALE_WORDS | {"and"}


def _split_into_sentences(text: str) -> list[str]:
    """Split text into sentences while keeping known abbreviations intact."""
    if not text:
        return []

    parts = _SENTENCE_BOUNDARY_RE.split(text)
    sentences: list[str] = []
    for part in parts:
        if not part:
            continue
        segment = part.strip()
        if not segment:
            continue
        if sentences and _ABBREVIATION_RE.search(sentences[-1]):
            sentences[-1] = f"{sentences[-1]} {segment}"
        else:
            sentences.append(segment)
    return sentences


def _strip_html_tags(text: str) -> str:
    """Remove HTML tags from text while preserving word boundaries."""
    if not text:
        return text
    return _HTML_TAG_RE.sub(" ", text)


def _title_starts_with_spelled_out_number(title: str | None) -> bool:
    """Return True if the provided title starts with spelled-out number words."""
    if not title:
        return False
    normalized = _normalize_whitespace(title)
    lowered = normalized.casefold()
    if lowered.startswith("chapter "):
        lowered = lowered[len("chapter ") :].lstrip()
    cleaned = re.sub(r"[^\w\s-]", " ", lowered)
    tokens = [token for token in re.split(r"[\s-]+", cleaned) if token]
    if not tokens:
        return False
    leading_tokens: list[str] = []
    for token in tokens:
        if token in _NUMBER_ALLOWED_TOKENS:
            leading_tokens.append(token)
        else:
            break
    if not leading_tokens:
        return False
    return any(token in _NUMBER_WORDS for token in leading_tokens)


def _replace_leading_numeric_marker(text: str) -> str:
    """Convert a leading Roman numeral or digit sequence to words."""
    if not text:
        return text
    match = _LEADING_CHAPTER_MARKER_RE.match(text)
    if not match:
        return text
    token = match.group("num")
    converted = replace_numeric_titles(token)
    if converted == token:
        return text
    leading_ws = match.group("space") or ""
    suffix = text[match.end() :]
    dot_match = re.match(r"^\.\s*", suffix)
    if dot_match:
        remainder = suffix[dot_match.end() :]
        whitespace = dot_match.group(0)[1:]
        suffix = f"{whitespace}{remainder}"
    return f"{leading_ws}{converted}{suffix}"


def _finalize_chapter_entry(
    chapter: dict[str, object],
    replace_number_titles: bool,
    chunk_size: int,
) -> None:
    """Apply post-processing to a single chapter entry."""
    source_title = chapter.pop("_source_title", None)
    display_title = chapter.get("chapter_title") or ""
    content_text = chapter.get("content_text") or ""

    if replace_number_titles:
        if display_title:
            updated_title = replace_numeric_titles(display_title)
            if (
                _title_starts_with_spelled_out_number(source_title)
                and not updated_title.casefold().startswith("chapter ")
            ):
                normalized_title = _normalize_whitespace(source_title)
                updated_title = f"CHAPTER {normalized_title.upper()}"
            display_title = updated_title
    insert_title = display_title
    if display_title and content_text and display_title[-1] not in ".!?":
        insert_title = f"{display_title}."
        content_text = insert_title + " "+ content_text

    chapter["chapter_title"] = display_title
    chapter["content_text"] = content_text
    chapter["chunks"] = _chunk_text(content_text, chunk_size)
    logger.debug(
        "Prepared chapter %s (%s) with %d chunks",
        chapter.get("chapter_number"),
        chapter["chapter_title"],
        len(chapter["chunks"]),
    )


def _split_long_segment(segment: str, limit: int) -> list[str]:
    """Split a long segment into smaller pieces that respect the size limit."""
    pieces: list[str] = []
    remainder = segment.strip()

    while remainder:
        if len(remainder) <= limit or limit <= 0:
            pieces.append(remainder)
            break

        split_index = remainder.rfind(" ", 0, limit + 1)
        if split_index <= 0:
            split_index = limit

        head = remainder[:split_index].rstrip()
        tail = remainder[split_index:].lstrip()

        if head:
            pieces.append(head)
        if not tail or tail == remainder:
            # Avoid infinite loops on strings without whitespace.
            if tail:
                pieces.append(tail[:limit])
                tail = tail[limit:]
            else:
                break
        remainder = tail

    return pieces


def _chunk_text(text: str, chunk_size: int) -> list[str]:
    """Split text into chunks of approximately chunk_size characters."""
    if not text:
        return []

    limit = max(chunk_size, 1)
    sentences = _split_into_sentences(text)
    segments: list[str] = []
    for sentence in sentences:
        segments.extend(_split_long_segment(sentence, limit))

    if not segments:
        return []

    chunks: list[str] = []
    current = ""

    for segment in segments:
        piece = segment.strip()
        if not piece:
            continue
        if not current:
            current = piece
            continue
        prospective_length = len(current) + 1 + len(piece)
        if prospective_length <= limit:
            current = f"{current} {piece}"
        else:
            chunks.append(current)
            current = piece

    if current:
        chunks.append(current)

    return chunks




def _build_href_lookup(book: epub.EpubBook) -> dict[str, epub.EpubItem]:
    """Create a lookup from normalized href to manifest items."""
    lookup: dict[str, epub.EpubItem] = {}
    for item in book.get_items():
        href = item.get_name()
        if not href:
            continue
        lookup[posixpath.normpath(href)] = item
    return lookup


def _flatten_toc_entries(entries: list) -> list[tuple[str, str]]:
    """Flatten a nested TOC structure into (title, href) pairs."""
    flattened: list[tuple[str, str]] = []
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


def _get_chapters_meta(book: epub.EpubBook) -> list[tuple[str, str]]:
    """Return chapter metadata using navigation data with spine fallback."""
    toc_entries = list(book.toc or [])
    chapters_meta = _flatten_toc_entries(toc_entries)
    chapters_meta = [(title, href) for title, href in chapters_meta if href]
    if not chapters_meta:
        logger.info("Falling back to OPF spine to determine chapter ordering.")
        chapters_meta = []
        for item_id, linear in book.spine:
            if (linear or "yes").lower() == "no":
                continue
            manifest_item = book.get_item_with_id(item_id)
            if not manifest_item:
                logger.debug("Skipping spine item missing from manifest: %s", item_id)
                continue
            href = manifest_item.get_name()
            if not href:
                logger.debug("Skipping spine item without href: %s", item_id)
                continue
            chapters_meta.append(("", href))
    if not chapters_meta:
        raise EPUBParsingError(
            "Unable to determine chapter documents from navigation data or spine.",
        )
    return chapters_meta


def _extract_cover_image(book: epub.EpubBook) -> tuple[str, str] | None:
    """Locate a cover image in the manifest and return its base64 representation and media type."""
    candidates: list[tuple[int, str, str, epub.EpubItem]] = []
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
        logger.info("Extracted cover image from %s", href)
        return encoded, media_type

    return None


def _tag_local_name(tag: str) -> str:
    """Return the local name component of an XML tag."""
    return tag.split("}")[-1]


def _get_metadata_attr(attrs: dict[str, str] | None, name: str) -> str | None:
    """Retrieve a metadata attribute value regardless of namespace prefix."""
    if not attrs:
        return None
    candidates = [
        name,
        name.lower(),
        name.upper(),
        f"opf:{name}",
        f"opf:{name.lower()}",
        f"opf:{name.upper()}",
        f"{{{OPF_NAMESPACE}}}{name}",
        f"{{{OPF_NAMESPACE}}}{name.lower()}",
        f"{{{OPF_NAMESPACE}}}{name.upper()}",
    ]
    for key in candidates:
        value = attrs.get(key)
        if value:
            return value

    lower_name = name.lower()
    for key, value in attrs.items():
        if not value:
            continue
        local = key
        if "}" in local:
            local = local.split("}")[-1]
        if ":" in local:
            local = local.split(":")[-1]
        if local.lower() == lower_name:
            return value
    return None


def _extract_text_from_xhtml(
    raw_bytes: bytes,
    ignored_classes: set[str] | None = None,
) -> str:
    """Convert an XHTML/HTML document to plain text."""
    ignored = {cls.casefold() for cls in ignored_classes} if ignored_classes else set()
    try:
        root = ET.parse(BytesIO(raw_bytes)).getroot()

        def _clear_text_content(element: ET.Element) -> None:
            element.text = ""
            element.tail = ""
            for child in list(element):
                _clear_text_content(child)

        heading_tags = {"h1", "h2", "h3", "h4", "h5"}
        for element in list(root.iter()):
            local_name = _tag_local_name(element.tag).lower()
            if local_name in {"title", "style"} or local_name in heading_tags:
                _clear_text_content(element)
                continue

            if ignored and element.attrib:
                class_attr = element.attrib.get("class", "")
                if class_attr:
                    class_tokens = {
                        token.casefold()
                        for token in class_attr.replace("\t", " ").split()
                        if token
                    }
                    if class_tokens & ignored:
                        _clear_text_content(element)
                        continue

        text_nodes = [_normalize_whitespace(node) for node in root.itertext()]
        filtered = [node for node in text_nodes if node]
        content = " ".join(filtered)
        content = html.unescape(content).replace("\u00a0", " ").replace("_", "")
        return _normalize_whitespace(content)
    except ET.ParseError:
        # Fallback: strip markup manually if the document is not clean XML.
        html_text = raw_bytes.decode("utf-8", errors="ignore")
        if ignored:
            class_alt = "|".join(re.escape(cls) for cls in sorted(ignored))
            html_text = re.sub(
                rf"<\s*([a-z0-9:_-]+)\b[^>]*class\s*=\s*['\"]([^'\"]*\b(?:{class_alt})\b[^'\"]*)['\"][^>]*>.*?<\s*/\s*\1\s*>",
                " ",
                html_text,
                flags=re.IGNORECASE | re.DOTALL,
            )
            html_text = re.sub(
                rf"<\s*([a-z0-9:_-]+)\b[^>]*class\s*=\s*['\"]([^'\"]*\b(?:{class_alt})\b[^'\"]*)['\"][^>]*/?>",
                " ",
                html_text,
                flags=re.IGNORECASE,
            )
        html_text = re.sub(
            r"<\s*(title|style|h[1-5])\b[^>]*>.*?<\s*/\s*\1\s*>",
            " ",
            html_text,
            flags=re.IGNORECASE | re.DOTALL,
        )
        text = re.sub(r"<[^>]+>", " ", html_text)
        text = html.unescape(text).replace("\u00a0", " ").replace("_", "")
        return _normalize_whitespace(text)


def _extract_chapter_text(
    raw_bytes: bytes,
    anchor: str | None,
    next_anchor: str | None = None,
    ignored_classes: set[str] | None = None,
) -> str:
    """Extract chapter text constrained by optional anchor boundaries."""
    if not anchor:
        return _extract_text_from_xhtml(raw_bytes, ignored_classes)

    html_text = raw_bytes.decode("utf-8", errors="ignore")
    anchor_pattern = re.compile(
        r'id\s*=\s*["\']' + re.escape(anchor) + r'["\']',
        re.IGNORECASE,
    )
    match = anchor_pattern.search(html_text)
    if not match:
        logger.warning(
            "Anchor '%s' not found in document; using full content.",
            anchor,
        )
        return _extract_text_from_xhtml(raw_bytes, ignored_classes)

    tag_start = html_text.rfind("<", 0, match.start())
    if tag_start == -1:
        tag_start = match.start()

    end_index = len(html_text)
    end_match: re.Match[str] | None = None
    if next_anchor:
        next_pattern = re.compile(
            r'id\s*=\s*["\']' + re.escape(next_anchor) + r'["\']',
            re.IGNORECASE,
        )
        end_match = next_pattern.search(html_text, match.end())
        if not end_match:
            logger.debug(
                "Next anchor '%s' was not found after '%s'; falling back to next id attribute.",
                next_anchor,
                anchor,
            )
            end_match = _ID_ATTRIBUTE_RE.search(html_text, match.end())
    if end_match:
        candidate_end = html_text.rfind("<", 0, end_match.start())
        end_index = candidate_end if candidate_end != -1 else end_match.start()

    snippet = html_text[tag_start:end_index]
    if not snippet.strip():
        return _extract_text_from_xhtml(raw_bytes, ignored_classes)

    wrapped = f"<div>{snippet}</div>".encode("utf-8", "ignore")
    return _extract_text_from_xhtml(wrapped, ignored_classes)


def _extract_metadata(book: epub.EpubBook) -> dict[str, object]:
    """Extract common metadata fields from the EbookLib representation."""

    def _metadata_values(namespace: str, name: str) -> list[tuple[str, dict[str, str]]]:
        entries: list[tuple[str, dict[str, str]]] = []
        try:
            raw_entries = book.get_metadata(namespace, name)
        except KeyError:
            return entries
        for value, attrs in raw_entries:
            text = _normalize_whitespace(value or "")
            if not text:
                continue
            entries.append((text, dict(attrs or {})))
        return entries

    metadata: dict[str, object] = {}
    refined_meta: dict[str, dict[str, str]] = {}
    standalone_meta: dict[str, str] = {}

    for text, attrs in _metadata_values("OPF", "meta"):
        property_name = _get_metadata_attr(attrs, "property") or _get_metadata_attr(attrs, "name")
        if not property_name:
            continue
        prop_key = property_name.split(":")[-1]
        refines = _get_metadata_attr(attrs, "refines")
        if refines:
            refined_meta.setdefault(refines.lstrip("#"), {})[prop_key] = text
        else:
            standalone_meta[prop_key] = text

    titles = [text for text, _ in _metadata_values("DC", "title")]
    if titles:
        metadata["title"] = titles[0]
        metadata["titles"] = titles

    creators: list[dict[str, str]] = []
    for text, attrs in _metadata_values("DC", "creator"):
        entry: dict[str, str] = {"name": text}
        role = _get_metadata_attr(attrs, "role")
        if role:
            entry["role"] = role
        file_as = _get_metadata_attr(attrs, "file-as")
        if file_as:
            entry["file_as"] = file_as
        creators.append(entry)
    if creators:
        metadata["creators"] = creators

    contributors: list[dict[str, str]] = []
    for text, attrs in _metadata_values("DC", "contributor"):
        entry: dict[str, str] = {"name": text}
        role = _get_metadata_attr(attrs, "role")
        if role:
            entry["role"] = role
        contributors.append(entry)
    if contributors:
        metadata["contributors"] = contributors

    subjects = [text for text, _ in _metadata_values("DC", "subject")]
    if subjects:
        metadata["subjects"] = subjects

    languages = [text for text, _ in _metadata_values("DC", "language")]
    if languages:
        metadata["languages"] = languages

    publishers = [text for text, _ in _metadata_values("DC", "publisher")]
    if publishers:
        metadata["publishers"] = publishers

    descriptions: list[str] = []
    for raw_text, _ in _metadata_values("DC", "description"):
        cleaned_description = _normalize_whitespace(
            _strip_html_tags(html.unescape(raw_text))
        )
        if cleaned_description:
            descriptions.append(cleaned_description)
    if descriptions:
        metadata["description"] = descriptions[0]
        if len(descriptions) > 1:
            metadata["descriptions"] = descriptions

    dates = [text for text, _ in _metadata_values("DC", "date")]
    if dates:
        metadata["date"] = dates[0]
        if len(dates) > 1:
            metadata["dates"] = dates

    rights_list = [text for text, _ in _metadata_values("DC", "rights")]
    if rights_list:
        metadata["rights"] = rights_list[0]
        if len(rights_list) > 1:
            metadata["rights_list"] = rights_list

    identifiers: list[dict[str, str]] = []
    for text, attrs in _metadata_values("DC", "identifier"):
        identifier_entry: dict[str, str] = {"value": text}
        identifier_id = attrs.get("id") or _get_metadata_attr(attrs, "id")
        if identifier_id:
            identifier_entry["id"] = identifier_id
        scheme = _get_metadata_attr(attrs, "scheme")
        if scheme:
            identifier_entry["scheme"] = scheme
        refined = refined_meta.get(identifier_id, {}) if identifier_id else {}
        identifier_type = (
            refined.get("identifier-type")
            or refined.get("type")
            or _get_metadata_attr(attrs, "identifier-type")
        )
        if identifier_type:
            identifier_entry["type"] = identifier_type
        identifiers.append(identifier_entry)
    if identifiers:
        metadata["identifiers"] = identifiers

    isbn_value = None
    for identifier in identifiers:
        scheme = identifier.get("scheme", "")
        id_type = identifier.get("type", "")
        value = identifier["value"]
        normalized = re.sub(r"[^0-9Xx]", "", value)
        if scheme and "isbn" in scheme.lower():
            isbn_value = value
            break
        if id_type and "isbn" in id_type.lower():
            isbn_value = value
            break
        if len(normalized) in (10, 13):
            isbn_value = value
            break
    if isbn_value:
        metadata["isbn"] = isbn_value

    for prop, value in standalone_meta.items():
        key = prop.split(":")[-1]
        if key not in metadata:
            metadata[key] = value

    return metadata


def parse_epub(
    epub_path: str,
    chunk_size: int = 2000,
    first_chapter_title: str | None = None,
    last_chapter_title: str | None = None,
    replace_number_titles: bool = False,
    ignore_classes: list[str] | None = None,
) -> dict[str, object]:
    """Parse an EPUB file and return metadata alongside chapter and chunk content."""
    logger.info("Parsing EPUB file: %s", epub_path)
    logger.debug("Using chunk size of %d characters", chunk_size)
    if first_chapter_title:
        logger.info("Filtering chapters from title: %s", first_chapter_title)
    if last_chapter_title:
        logger.info("Applying last chapter filter: %s", last_chapter_title)
    ignored_classes_set: set[str] | None = None
    if ignore_classes:
        processed = {
            cls.casefold()
            for cls in (entry.strip() for entry in ignore_classes)
            if cls
        }
        if processed:
            ignored_classes_set = processed

    cover_image_data: str | None = None
    cover_image_media_type: str | None = None

    try:
        book = epub.read_epub(epub_path, options={"ignore_ncx": False})
    except FileNotFoundError:
        raise
    except epub.EpubException as exc:
        raise EPUBParsingError(f"Failed to load EPUB structure: {exc}") from exc
    except Exception as exc:
        raise EPUBParsingError(f"Failed to read EPUB: {exc}") from exc

    metadata = _extract_metadata(book)
    cover_image_info = _extract_cover_image(book)
    if cover_image_info:
        cover_image_data, cover_image_media_type = cover_image_info

    chapters_meta = _get_chapters_meta(book)
    href_lookup = _build_href_lookup(book)

    chapter_refs: list[tuple[str, str, str | None, str]] = []
    for title, src in chapters_meta:
        if not src:
            continue
        if "#" in src:
            relative_path, fragment = src.split("#", maxsplit=1)
        else:
            relative_path, fragment = src, None
        normalized_relative = posixpath.normpath(relative_path)
        chapter_refs.append((title, normalized_relative, fragment, relative_path))

    chapters = []
    seen_entries: set[tuple[str, str | None]] = set()

    for index, (title, normalized_relative, fragment, original_path) in enumerate(chapter_refs, start=1):
        entry_key = (normalized_relative, fragment)
        if entry_key in seen_entries:
            suffix = f"#{fragment}" if fragment else ""
            logger.debug("Skipping duplicate chapter source: %s%s", normalized_relative, suffix)
            continue
        seen_entries.add(entry_key)

        next_fragment: str | None = None
        if index < len(chapter_refs):
            _, next_normalized, next_anchor, _ = chapter_refs[index]
            if next_normalized == normalized_relative and next_anchor:
                next_fragment = next_anchor

        item = href_lookup.get(normalized_relative)
        if item is None:
            item = book.get_item_with_href(original_path)
        if item is None and normalized_relative != original_path:
            item = book.get_item_with_href(normalized_relative)

        if item is None:
            logger.warning("Chapter resource missing in manifest: %s", normalized_relative)
            continue

        if item.get_type() != ebooklib.ITEM_DOCUMENT:
            logger.debug("Skipping non-document item in navigation: %s", normalized_relative)
            continue
        is_chapter = getattr(item, "is_chapter", None)
        if callable(is_chapter) and not is_chapter():
            logger.debug("Skipping non-chapter document in navigation: %s", normalized_relative)
            continue

        raw_content = item.get_content()
        if isinstance(raw_content, str):
            raw_content = raw_content.encode("utf-8", "ignore")

        chapter_number = len(chapters) + 1
        content_text = _extract_chapter_text(
            raw_content,
            fragment,
            next_fragment,
            ignored_classes_set,
        )
        content_text = _dedupe_leading_title(title, content_text)
        display_title = title or f"Chapter {chapter_number}"
        chapter_entry = {
            "chapter_number": chapter_number,
            "chapter_title": display_title,
            "content_text": content_text,
            "_source_title": title,
        }
        chapters.append(chapter_entry)

    for chapter in chapters:
        _finalize_chapter_entry(chapter, replace_number_titles, chunk_size)

    modified = False
    if first_chapter_title:
        normalized_target = _normalize_whitespace(first_chapter_title).casefold()
        match_index = None
        for idx, chapter in enumerate(chapters):
            chapter_title = _normalize_whitespace(chapter.get("chapter_title", ""))
            if chapter_title and chapter_title.casefold() == normalized_target:
                match_index = idx
                break

        if match_index is not None:
            if match_index > 0:
                logger.info(
                    "Dropping %d chapter(s) prior to '%s'.",
                    match_index,
                    first_chapter_title,
                )
            chapters = chapters[match_index:]
            modified = True
        else:
            logger.warning(
                "Requested first chapter title '%s' was not found; returning all chapters.",
                first_chapter_title,
            )
    if last_chapter_title:
        normalized_target = _normalize_whitespace(last_chapter_title).casefold()
        match_index = None
        for idx, chapter in enumerate(chapters):
            chapter_title = _normalize_whitespace(chapter.get("chapter_title", ""))
            if chapter_title and chapter_title.casefold() == normalized_target:
                match_index = idx
                break
        if match_index is not None and match_index + 1 < len(chapters):
            dropped = len(chapters) - (match_index + 1)
            chapters = chapters[: match_index + 1]
            logger.info(
                "Dropping %d chapter(s) after '%s'.",
                dropped,
                last_chapter_title,
            )
            modified = True
        elif match_index is None:
            logger.warning(
                "Requested last chapter title '%s' was not found; returning all chapters.",
                last_chapter_title,
            )

    if modified:
        for new_number, chapter in enumerate(chapters, start=1):
            chapter["chapter_number"] = new_number

    result = {**metadata, "chapters": chapters}
    if cover_image_data:
        result["cover_image"] = cover_image_data
        if cover_image_media_type:
            result["cover_image_media_type"] = cover_image_media_type
    logger.info(
        "Parsed %d chapters with metadata keys: %s",
        len(chapters),
        ", ".join(sorted(metadata)) if metadata else "none",
    )
    return {key: value for key, value in result.items() if value not in (None, [], {}, "")}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Parse an EPUB file into a JSON object with metadata and chapters.",
    )
    parser.add_argument("epub", help="Path to the EPUB file to parse.")
    parser.add_argument(
        "-o",
        "--output",
        help="Destination path for the resulting JSON. Defaults to stdout.",
    )
    parser.add_argument(
        "--pretty",
        default=True,
        action="store_true",
        help="Pretty-print the JSON output with indentation.",
    )
    parser.add_argument(
        "--first-chapter-title",
        help="Skip chapters until this title is encountered and renumber from one.",
    )
    parser.add_argument(
        "--last-chapter-title",
        help="Stop after this chapter title, removing any subsequent chapters.",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=2000,
        help="Maximum number of characters per chapter chunk (default: 2000).",
    )
    parser.add_argument(
        "--number-title-to-english",
        action="store_true",
        help="Replace Roman numerals and digits in chapter titles with their English word equivalents.",
    )
    parser.add_argument(
        "--ignore-class",
        help="Comma-separated list of CSS class names to ignore when extracting content.",
    )

    args = parser.parse_args(argv)

    ignore_classes: list[str] | None = None
    if args.ignore_class:
        ignore_classes = [
            entry.strip()
            for entry in args.ignore_class.split(",")
            if entry and entry.strip()
        ]

    try:
        epub_data = parse_epub(
            args.epub,
            chunk_size=args.chunk_size,
            first_chapter_title=args.first_chapter_title,
            last_chapter_title=args.last_chapter_title,
            replace_number_titles=args.number_title_to_english,
            ignore_classes=ignore_classes,
        )
    except EPUBParsingError as exc:
        parser.error(str(exc))
        return 2
    except FileNotFoundError:
        parser.error(f"EPUB file not found: {args.epub}")
        return 2

    dump_kwargs = {"ensure_ascii": False}
    if args.pretty:
        dump_kwargs["indent"] = 2

    if args.output:
        with open(args.output, "w", encoding="utf-8") as handle:
            json.dump(epub_data, handle, **dump_kwargs)
    else:
        json.dump(epub_data, sys.stdout, **dump_kwargs)
        sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
