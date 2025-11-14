from __future__ import annotations

import argparse
import base64
import datetime as dt
import json
import mimetypes
import os
import re
import tempfile
import uuid
from email.utils import format_datetime
from pathlib import Path
from typing import Any
from urllib.parse import quote
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from html import unescape

from dotenv import load_dotenv


def upload_feed(metadata: dict, feedFile: str) -> None:
    """Upload the generated feed XML and chapter MP3 files to the remote host."""
    # Lazy import to avoid altering module load order
    from scp import run_scp, run_ssh  # type: ignore

    load_dotenv()

    remote_base = os.getenv("REMOTE_BASE")
    if not remote_base:
        raise RuntimeError("Environment variable REMOTE_BASE is required.")

    # Determine remote host (hostname[:port])
    remote = os.getenv("SCP_REMOTE") or os.getenv("REMOTE_HOST")
    if not remote:
        # Fallback: derive host from AUDIO_URL_PREFIX
        try:
            parsed = urllib.parse.urlparse(os.getenv(ENV_AUDIO_URL_PREFIX, ""))
            remote = parsed.hostname or ""
        except Exception:
            remote = ""
    if not remote:
        raise RuntimeError("Remote host not set. Define SCP_REMOTE or REMOTE_HOST or set a valid AUDIO_URL_PREFIX.")

    feed_path = Path(feedFile).expanduser()
    if not feed_path.is_file():
        raise FileNotFoundError(f"Feed file not found: {feed_path}")

    # Read channel title from the generated feed XML
    try:
        tree = ET.parse(feed_path)
        root = tree.getroot()
        channel = root.find("channel") if root is not None else None
        title_el = channel.find("title") if channel is not None else None
        channel_title = (title_el.text or "").strip() if title_el is not None else ""
    except Exception as exc:
        raise RuntimeError(f"Failed to parse feed file '{feedFile}': {exc}")

    if not channel_title:
        # Fallback to metadata title if channel title missing
        channel_title = (metadata.get("title") or "").strip()
    if not channel_title:
        raise RuntimeError("Channel title not found in feed or metadata.")

    # Replace spaces with underscores for the remote directory name
    safe_title = re.sub(r"\s+", "_", channel_title).strip("_") or "feed"
    safe_title = safe_title.lower()
    remote_dir = f"{remote_base.rstrip('/')}/{safe_title}"

    # Create directory on remote (use -p to create parents if needed)
    # Quote the path to handle special characters safely
    cmd = f"mkdir -p '{remote_dir}'"
    run_ssh(remote, cmd)

    # Determine where MP3 files live locally
    audio_dir_value = (
        metadata.get("audio_dir")
        or metadata.get("audio_directory")
        or metadata.get("audio_path")
        or metadata.get("audio_base")
        or metadata.get("audio_output_dir")
    )
    if not audio_dir_value:
        raise RuntimeError(
            "Audio directory not supplied. Set metadata['audio_dir'] (or audio_directory/audio_path/audio_base/audio_output_dir)."
        )

    audio_base_dir = Path(str(audio_dir_value)).expanduser()
    if not audio_base_dir.exists():
        raise FileNotFoundError(f"Audio directory not found: {audio_base_dir}")
    if not audio_base_dir.is_dir():
        raise NotADirectoryError(f"Audio directory path is not a directory: {audio_base_dir}")

    candidate_dir: Path | None = None
    direct_match = audio_base_dir / safe_title
    if direct_match.is_dir():
        candidate_dir = direct_match
    elif audio_base_dir.name.lower() == safe_title.lower():
        candidate_dir = audio_base_dir
    elif any(audio_base_dir.glob("*.mp3")):
        candidate_dir = audio_base_dir
    else:
        try:
            for entry in audio_base_dir.iterdir():
                if entry.is_dir() and entry.name.lower() == safe_title.lower():
                    candidate_dir = entry
                    break
        except PermissionError as exc:  # pragma: no cover - filesystem dependent
            raise RuntimeError(f"Unable to inspect audio directory '{audio_base_dir}': {exc}") from exc

    if candidate_dir is None:
        raise FileNotFoundError(
            f"Audio directory for '{channel_title}' not found under {audio_base_dir}. Expected folder '{safe_title}'."
        )

    mp3_files = sorted(path for path in candidate_dir.rglob("*.mp3") if path.is_file())
    if not mp3_files:
        raise FileNotFoundError(f"No MP3 files found in {candidate_dir}.")

    # Upload feed XML
    run_scp(feed_path, f"{remote_dir}/feed.xml", remote)

    # Upload cover image if present
    cover_filename = metadata.get("cover_image_file")
    cover_path: Path | None = None
    if cover_filename:
        potential = candidate_dir / cover_filename
        if potential.is_file():
            cover_path = potential
    if cover_path is None:
        fallback = candidate_dir / "cover.jpg"
        if fallback.is_file():
            cover_path = fallback
        else:
            alternatives = sorted(candidate_dir.glob("cover.*"))
            if alternatives:
                cover_path = alternatives[0]
    if cover_path is not None and cover_path.is_file():
        run_scp(cover_path, f"{remote_dir}/{cover_path.name}", remote)

    # Upload each MP3 into the channel directory
    for mp3_file in mp3_files:
        run_scp(mp3_file, f"{remote_dir}/{mp3_file.name}", remote)


def generate_cover(feed_xml: str, output_image: str) -> None:
    """Generate a cover image using the OpenRouter nano-banana model."""
    load_dotenv()

    feed_path = Path(feed_xml).expanduser()
    if not feed_path.is_file():
        raise FileNotFoundError(f"Feed XML not found: {feed_path}")

    try:
        tree = ET.parse(feed_path)
    except ET.ParseError as exc:
        raise RuntimeError(f"Failed to parse feed file '{feed_xml}': {exc}") from exc

    root = tree.getroot()
    channel = root.find("channel")
    if channel is None:
        raise RuntimeError("Feed does not contain a channel element.")

    book_name = (channel.findtext("title") or "").strip()
    author = (channel.findtext(f"{{{ITUNES_NS}}}author") or "").strip()

    if not book_name:
        raise RuntimeError("channel/title element is missing or empty in the feed XML.")
    if not author:
        raise RuntimeError("channel/itunes:author element is missing or empty in the feed XML.")

    api_key = os.getenv("OPENROUTE_API_KEY") or os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError("Missing API key. Set OPENROUTE_API_KEY or OPENROUTER_API_KEY in the environment.")

    model_name = os.getenv("OPENROUTE_MODEL", "google/gemini-2.5-flash-image-preview")
    prompt = f"generate a cover page for {author}'s book {book_name}, w/h ratio 3:4. no background."
    request_payload = {
        "model": model_name,
        "messages": [
            {
                "role": "user",
                "content": prompt,
            }
        ],
        "modalities": ["image", "text"],
        "image_config": {"aspect_ratio": "3:4"},
    }
    request_payload_text = json.dumps(request_payload, separators=(",", ":"), ensure_ascii=False)
    print("OpenRouter request payload:")
    print(request_payload_text)

    request_bytes = request_payload_text.encode("utf-8")
    req = urllib.request.Request("https://openrouter.ai/api/v1/chat/completions", data=request_bytes, method="POST")
    req.add_header("Content-Type", "application/json")
    req.add_header("Accept", "application/json")
    req.add_header("Authorization", f"Bearer {api_key}")
    req.add_header("HTTP-Referer", os.getenv("OPENROUTE_HTTP_REFERER", "https://github.com/ai-pod/ai-pod"))
    req.add_header("X-Title", os.getenv("OPENROUTE_HTTP_TITLE", "AI Pod Cover Generator"))

    response_text: str = ""
    content_type = ""
    response_bytes: bytes = b""
    try:
        with urllib.request.urlopen(req, timeout=120) as response:
            response_bytes = response.read()
            content_type = response.headers.get("Content-Type", "")
            if content_type.startswith("image/"):
                response_preview = base64.b64encode(response_bytes).decode("ascii")
                print("OpenRouter response (image, base64 encoded preview):")
                print(response_preview[:400])
            else:
                response_text_preview = response_bytes.decode("utf-8", errors="replace")
                print("OpenRouter response text:")
                print(response_text_preview)
                lowered = response_text_preview.lower()
                if "<html" in lowered and "model not found" in lowered:
                    raise RuntimeError(
                        f"OpenRouter reported that model '{model_name}' was not found. "
                        "Verify the model slug or request access to it."
                    )
                if "<html" in lowered:
                    raise RuntimeError(
                        "Unexpected HTML response from OpenRouter image API. "
                        "Check API key permissions and endpoint availability."
                    )
                response_text = response_text_preview
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="ignore")
        raise RuntimeError(f"Cover generation request failed with status {exc.code}: {body}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Error contacting OpenRouter image API: {exc.reason}") from exc

    image_bytes: bytes | None = None
    if content_type.startswith("image/"):
        image_bytes = response_bytes
    else:
        if not response_text.strip():
            raise RuntimeError("OpenRouter image API returned an empty response.")
        try:
            payload = json.loads(response_text)
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"Invalid JSON from OpenRouter image API: {exc}\nResponse: {response_text[:200]}") from exc

        def _decode_data_url(value: Any) -> bytes | None:
            if not isinstance(value, str):
                return None
            if value.startswith("data:"):
                _, _, encoded = value.partition(",")
                if encoded:
                    try:
                        return base64.b64decode(encoded, validate=True)
                    except (ValueError, TypeError):
                        return None
                return None
            try:
                return base64.b64decode(value, validate=True)
            except (ValueError, TypeError):
                return None

        def _extract_from_image_entry(image_entry: Any) -> bytes | None:
            if not isinstance(image_entry, dict):
                return None
            url_info = image_entry.get("image_url")
            if isinstance(url_info, dict):
                data = _decode_data_url(url_info.get("url"))
                if data:
                    return data
            for key in ("url", "image_url", "image"):
                data = _decode_data_url(image_entry.get(key))
                if data:
                    return data
            for key in ("image_base64", "b64_json", "image_b64", "base64"):
                if key in image_entry:
                    data = _decode_data_url(image_entry.get(key))
                    if data:
                        return data
            return None

        def _extract_from_content(content: Any) -> bytes | None:
            if isinstance(content, list):
                for part in content:
                    if isinstance(part, dict):
                        for key in ("image_base64", "b64_json", "image_b64", "base64"):
                            data = _decode_data_url(part.get(key))
                            if data:
                                return data
                        data = _decode_data_url(part.get("image_url"))
                        if data:
                            return data
                        data = _decode_data_url(part.get("url"))
                        if data:
                            return data
                        data = _decode_data_url(part.get("text"))
                        if data:
                            return data
                    else:
                        data = _decode_data_url(part)
                        if data:
                            return data
            elif isinstance(content, dict):
                return _extract_from_image_entry(content)
            else:
                return _decode_data_url(content)
            return None

        def _extract_image_bytes(data: dict[str, Any]) -> bytes | None:
            choices = data.get("choices")
            if isinstance(choices, list):
                for choice in choices:
                    if not isinstance(choice, dict):
                        continue
                    message = choice.get("message")
                    if isinstance(message, dict):
                        images = message.get("images")
                        if isinstance(images, list):
                            for image in images:
                                data_bytes = _extract_from_image_entry(image)
                                if data_bytes:
                                    return data_bytes
                        content = message.get("content")
                        data_bytes = _extract_from_content(content)
                        if data_bytes:
                            return data_bytes
                        data_bytes = _decode_data_url(message.get("text"))
                        if data_bytes:
                            return data_bytes
            images = data.get("images")
            if isinstance(images, list):
                for image in images:
                    data_bytes = _extract_from_image_entry(image)
                    if data_bytes:
                        return data_bytes
            # Some responses may return top-level data URLs
            return _extract_from_content(data.get("content"))

        image_bytes = _extract_image_bytes(payload)

    if not image_bytes:
        raise RuntimeError("No image bytes were produced by the OpenRouter image API.")

    output_path = Path(output_image).expanduser()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(image_bytes)


ITUNES_NS = "http://www.itunes.com/dtds/podcast-1.0.dtd"
ATOM_NS = "http://www.w3.org/2005/Atom"

ET.register_namespace("itunes", ITUNES_NS)
ET.register_namespace("atom", ATOM_NS)

ENV_AUDIO_URL_PREFIX = "AUDIO_URL_PREFIX"
ENV_OLLAMA_API_URL = "OLLAMA_API_URL"
OLLAMA_MODEL = "gemma:7b"

try:
    from mutagen import File as MutagenFile
except ImportError:  # pragma: no cover - optional dependency
    MutagenFile = None  # type: ignore[assignment]


def _indent(element: ET.Element, level: int = 0) -> None:
    """Pretty-print helper to add line breaks/indentation."""
    indent_str = "\n" + "  " * level
    children = list(element)
    if not children:
        if level and (not element.tail or not element.tail.strip()):
            element.tail = indent_str
        return

    if not element.text or not element.text.strip():
        element.text = indent_str + "  "
    for child in children:
        _indent(child, level + 1)
        if not child.tail or not child.tail.strip():
            child.tail = indent_str + "  "
    if not children[-1].tail or not children[-1].tail.strip():
        children[-1].tail = indent_str


def _format_duration(total_seconds: float | None) -> str | None:
    if total_seconds is None:
        return None
    total = int(round(total_seconds))
    hours, remainder = divmod(total, 3600)
    minutes, seconds = divmod(remainder, 60)
    if hours > 0:
        return f"{hours}:{minutes:02d}:{seconds:02d}"
    return f"{minutes:02d}:{seconds:02d}"

def _strip_html(text: str | None) -> str:
    if not text:
        return ""
    unescaped = unescape(text)
    unescaped = unescaped.replace("\xa0", " ")
    without_tags = re.sub(r"<[^>]+>", "", unescaped)
    cleaned = re.sub(r"&[A-Za-z0-9#]+;", " ", without_tags)
    return re.sub(r"\s+", " ", cleaned).strip()

def summarize_chapter_text(
    text: str,
    *,
    api_url: str,
    model: str = OLLAMA_MODEL,
    max_characters: int = 1000,
) -> str:
    """Summarize chapter text using a local Ollama instance."""
    cleaned_text = (text or "").strip()
    if not cleaned_text:
        return ""

    endpoint = urllib.parse.urljoin(api_url.rstrip("/") + "/", "api/generate")
    constraint=""
    if max_characters > 0:
        constraint = f"Keep the summary under {max_characters} characters. keep it general. generate one paragraph with 3-4 sentences only. don't include any * character in the response."
        
    else:
        constraint = "Use 3-4 sentences, highlight key events, and avoid spoilers.highlighting key events and avoiding spoilers. generate one paragraph with 3-4 sentences only, don't include any * character in the response."
        
    prompt = f"Summarize the following book chapter into a concise podcast episode description. f{constraint}\n\n{cleaned_text}"
    #print(f"Prompt: {prompt}")
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": 0.2, "num_predict": max_characters},
    }

    data = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        endpoint,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            body = response.read()
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Failed to reach Ollama API: {exc}") from exc

    try:
        result = json.loads(body.decode("utf-8"))
    except json.JSONDecodeError as exc:
        raise RuntimeError("Received invalid JSON from the Ollama API.") from exc

    summary = (result.get("response") or "").strip()
    if not summary:
        raise RuntimeError("Ollama API returned an empty summary response.")
    return summary

def _slugify(text: str) -> str:
    text = text.strip()
    if not text:
        return "chapter"
    cleaned = re.sub(r"[^A-Za-z0-9]+", "_", text)
    cleaned = cleaned.strip("_")
    return cleaned or "chapter"


def _first(values: list[Any] | None) -> Any | None:
    if not values:
        return None
    return next((value for value in values if value), None)


def _parse_iso_datetime(value: str | None) -> dt.datetime | None:
    if not value:
        return None
    value = value.strip()
    if not value:
        return None
    replacement = value.replace("Z", "+00:00")
    try:
        return dt.datetime.fromisoformat(replacement)
    except ValueError:
        return None


def _as_pubdate(date_value: dt.datetime | None, fallback: dt.datetime) -> str:
    moment = date_value or fallback
    if moment.tzinfo is None:
        moment = moment.replace(tzinfo=dt.timezone.utc)
    return format_datetime(moment)


def _chapter_filename(chapter: dict[str, Any]) -> str:
    number = chapter.get("chapter_number") or 0
    try:
        number = int(number)
    except (ValueError, TypeError):
        number = 0
    title = chapter.get("chapter_title") or f"Chapter {number or ''}".strip()
    slug = _slugify(title)
    return f"{number:03d}_{slug}.mp3"


def _build_item(
    chapter: dict[str, Any],
    audio_dir: Path,
    audio_base_url: str | None,
    default_author: str,
    item_pub_date: dt.datetime,
    book_title: str | None,
) -> ET.Element:
    title = chapter.get("chapter_title") or f"Chapter {chapter.get('chapter_number', '')}".strip()
    filename = chapter.get("audio_file") or _chapter_filename(chapter)
    file_path = audio_dir / filename
    if not file_path.is_file():
        raise FileNotFoundError(f"Audio file not found for chapter '{title}': {file_path}")

    file_size = file_path.stat().st_size
    if audio_base_url:
        base = audio_base_url.rstrip("/")
        title_slug = _slugify(book_title or "")
        if not title_slug:
            title_slug = "book"
        url = f"{base}/{quote(title_slug.lower())}/{quote(filename)}"
    else:
        url = file_path.resolve().as_uri()

    guid_value = str(uuid.uuid4())
    metadata_title = (book_title or "").strip()
    chapter_number = chapter.get("chapter_number")
    chapter_label = "chapter"
    if chapter_number not in (None, ""):
        chapter_label = f"chapter {chapter_number}"
    if metadata_title:
        description_text = f"{chapter_label} of {metadata_title}"
    else:
        description_text = chapter_label
    summary = description_text[:5000] if description_text else ""

    item = ET.Element("item")
    ET.SubElement(item, "title").text = title
    ET.SubElement(item, "guid").text = guid_value
    ET.SubElement(item, "link").text = url
    ET.SubElement(item, "pubDate").text = _as_pubdate(item_pub_date, item_pub_date)
    if summary:
        ET.SubElement(item, "description").text = summary
        ET.SubElement(item, f"{{{ITUNES_NS}}}summary").text = summary

    ET.SubElement(item, f"{{{ITUNES_NS}}}episode").text = str(chapter.get("chapter_number", ""))
    ET.SubElement(item, f"{{{ITUNES_NS}}}episodeType").text = "full"
    ET.SubElement(item, f"{{{ITUNES_NS}}}author").text = chapter.get("author") or default_author

    enclosure = ET.SubElement(item, "enclosure")
    enclosure.set("url", url)
    enclosure.set("type", "audio/mpeg")
    enclosure.set("length", str(file_size))

    if MutagenFile is None:
        raise RuntimeError(
            "mutagen is required to compute audio durations. Install it with 'pip install mutagen'."
        )
    audio_meta = MutagenFile(file_path)
    duration_text = _format_duration(getattr(getattr(audio_meta, "info", None), "length", None))
    if duration_text:
        ET.SubElement(item, f"{{{ITUNES_NS}}}duration").text = duration_text

    return item


def build_feed(
    metadata: dict[str, Any],
    audio_dir: Path,
    *,
    audio_base_url: str | None,
    feed_title: str | None,
    feed_description: str | None,
    feed_language: str | None,
    author_override: str | None,
    site_url: str | None,
    image_url: str | None,
    explicit: bool,
    ollama_api_url: str,
) -> ET.ElementTree:
    channel_title = feed_title or metadata.get("title") or _first(metadata.get("titles")) or "Untitled Podcast"
    raw_description = (
        feed_description
        or metadata.get("description")
        or _first(metadata.get("descriptions"))
        or f"Podcast adaptation of {channel_title}."
    )
    description = _strip_html(raw_description)
    if not description:
        description = _strip_html(f"Podcast adaptation of {channel_title}.")
    language = feed_language or _first(metadata.get("languages")) or "en"
    default_author = (
        author_override
        or _first([creator.get("name") for creator in metadata.get("creators", []) if isinstance(creator, dict)])
        or "Unknown"
    )
    pub_date = _parse_iso_datetime(metadata.get("date")) or _parse_iso_datetime(metadata.get("modified"))
    now = dt.datetime.now(dt.timezone.utc)
    channel_pub_date = _as_pubdate(pub_date, now)
    base_pub_date = pub_date or now
    safe_title = re.sub(r"\s+", "_", channel_title).strip("_") or "feed"
    safe_title = safe_title.lower()
    rss = ET.Element("rss", attrib={"version": "2.0"})
    channel = ET.SubElement(rss, "channel")
    ET.SubElement(channel, "title").text = channel_title
    ET.SubElement(channel, "description").text = description
    ET.SubElement(channel, "language").text = language
    ET.SubElement(channel, "pubDate").text = channel_pub_date
    ET.SubElement(channel, "lastBuildDate").text = _as_pubdate(now, now)
    ET.SubElement(channel, f"{{{ITUNES_NS}}}author").text = default_author
    ET.SubElement(channel, f"{{{ITUNES_NS}}}summary").text = description
    ET.SubElement(channel, f"{{{ITUNES_NS}}}explicit").text = "yes" if explicit else "no"

    if site_url or audio_base_url:
        ET.SubElement(channel, "link").text = site_url or audio_base_url
        atom_self = ET.SubElement(channel, f"{{{ATOM_NS}}}link")
        atom_self.set("href", site_url or audio_base_url or "")
        atom_self.set("rel", "self")
        atom_self.set("type", "application/rss+xml")

    cover_image_url = image_url
    cover_filename: str | None = None
    cover_written = False
    metadata_cover_b64 = metadata.get("cover_image")
    metadata_cover_type = metadata.get("cover_image_media_type")

    if metadata_cover_b64 and metadata_cover_type:
        cover_dir = audio_dir
        cover_dir.mkdir(parents=True, exist_ok=True)
        media_type_lower = str(metadata_cover_type).lower()
        extension = {
            "image/jpeg": ".jpg",
            "image/jpg": ".jpg",
            "image/png": ".png",
            "image/webp": ".webp",
        }.get(media_type_lower, mimetypes.guess_extension(media_type_lower) or ".img")
        if not extension.startswith("."):
            extension = f".{extension}"
        cover_filename = f"cover{extension}"
        cover_path = cover_dir / cover_filename
        try:
            try:
                image_bytes = base64.b64decode(metadata_cover_b64, validate=True)  # type: ignore[arg-type]
            except TypeError:
                image_bytes = base64.b64decode(metadata_cover_b64)
            if not image_bytes:
                raise ValueError("decoded image is empty")
            cover_path.write_bytes(image_bytes)
        except Exception as exc:
            print(f"Warning: failed to decode cover image from metadata ({exc}); falling back to generated cover.")
            if cover_path.exists():
                try:
                    cover_path.unlink()
                except OSError:
                    pass
            cover_filename = None
        else:
            cover_written = True
            metadata["cover_image_file"] = cover_filename
            if audio_base_url:
                cover_image_url = f"{audio_base_url.rstrip('/')}/{safe_title}/{cover_filename}"
            else:
                cover_image_url = cover_path.resolve().as_uri()

    if not cover_written and audio_base_url:
        cover_dir = audio_dir
        cover_dir.mkdir(parents=True, exist_ok=True)
        cover_filename = "cover.jpg"
        cover_path = cover_dir / cover_filename
        temp_path: Path | None = None
        try:
            with tempfile.NamedTemporaryFile("wb", suffix=".xml", delete=False) as temp_feed:
                temp_path = Path(temp_feed.name)
                temp_tree = ET.ElementTree(rss)
                temp_tree.write(temp_feed, encoding="utf-8", xml_declaration=True)
            generate_cover(str(temp_path), str(cover_path))
        finally:
            if temp_path is not None:
                try:
                    temp_path.unlink()
                except OSError:
                    pass
        metadata["cover_image_file"] = cover_filename
        cover_image_url = f"{audio_base_url.rstrip('/')}/{safe_title}/{cover_filename}"

    if cover_image_url:
        image = ET.SubElement(channel, "image")
        ET.SubElement(image, "url").text = cover_image_url
        ET.SubElement(image, "title").text = channel_title
        ET.SubElement(image, "link").text = site_url or audio_base_url or ""
        ET.SubElement(channel, f"{{{ITUNES_NS}}}image").set("href", cover_image_url)

    chapters = metadata.get("chapters") or []
    if not chapters:
        raise ValueError("No chapters found in metadata JSON.")

    for offset, chapter in enumerate(chapters):
        print(f"processing Chapter {chapter.get('chapter_number', '')}: {chapter.get('chapter_title', '')}")
        chapter_pub = base_pub_date + dt.timedelta(days=offset)
        item = _build_item(
            chapter,
            audio_dir,
            audio_base_url,
            default_author,
            chapter_pub,
            metadata.get("title"),
        )
        channel.append(item)

    return ET.ElementTree(rss)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate a podcast RSS feed from EPUB parsing output and audio files. "
            f"Requires {ENV_AUDIO_URL_PREFIX} and {ENV_OLLAMA_API_URL} environment variables."
        ),
    )
    parser.add_argument(
        "book_title",
        help="Book title used to locate ./data/{book_title}/book.json, book.xml, and audio output.",
    )
    parser.add_argument("--feed-title", help="Custom podcast title. Defaults to the EPUB metadata title.")
    parser.add_argument("--feed-description", help="Custom podcast description.")
    parser.add_argument("--feed-language", help="Override the podcast language (e.g., 'en').")
    parser.add_argument("--author", help="Override the podcast author name.")
    parser.add_argument(
        "--site-url",
        help=f"Website or landing page for the podcast (defaults to {ENV_AUDIO_URL_PREFIX} if unset).",
    )
    parser.add_argument("--image", help="URL to cover art image (square, at least 1400x1400).")
    parser.add_argument(
        "--explicit",
        action="store_true",
        help="Mark the podcast as explicit for the itunes:explicit tag.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    load_dotenv()
    args = parse_args(argv)

    base_dir = Path("data") / args.book_title
    metadata_path = base_dir / "book.json"
    output_path = base_dir / "book.xml"
    audio_dir = base_dir / "kokoro"

    if not metadata_path.is_file():
        raise FileNotFoundError(f"Metadata JSON not found: {metadata_path}")
    if not audio_dir.is_dir():
        raise NotADirectoryError(f"Audio directory is not a directory: {audio_dir}")

    audio_url_prefix = os.getenv(ENV_AUDIO_URL_PREFIX)
    if not audio_url_prefix:
        raise RuntimeError(
            f"Environment variable {ENV_AUDIO_URL_PREFIX} is required to publish audio file URLs."
        )
    ollama_api_url = os.getenv(ENV_OLLAMA_API_URL)
    if not ollama_api_url:
        raise RuntimeError(
            f"Environment variable {ENV_OLLAMA_API_URL} is required to summarize chapter content."
        )

    with metadata_path.open("r", encoding="utf-8") as handle:
        metadata = json.load(handle)

    metadata["audio_dir"] = str(audio_dir)

    feed = build_feed(
        metadata,
        audio_dir,
        audio_base_url=audio_url_prefix,
        feed_title=args.feed_title,
        feed_description=args.feed_description,
        feed_language=args.feed_language,
        author_override=args.author,
        site_url=args.site_url or audio_url_prefix,
        image_url=args.image,
        explicit=args.explicit,
        ollama_api_url=ollama_api_url,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    _indent(feed.getroot())
    feed.write(output_path, encoding="utf-8", xml_declaration=True)
    print(f"Generated RSS feed at {output_path}")

    upload_feed(metadata, str(output_path))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
