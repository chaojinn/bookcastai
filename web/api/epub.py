from __future__ import annotations

import json
import logging
import os
from pathlib import Path

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException
from fastapi.responses import Response
from supertokens_python.recipe.session import SessionContainer
from supertokens_python.recipe.session.framework.fastapi import verify_session

logger = logging.getLogger(__name__)

router = APIRouter()


def _get_base_dir() -> Path:
    base_dir = os.getenv("PODS_BASE")
    if not base_dir:
        raise HTTPException(status_code=500, detail="PODS_BASE is not configured.")
    return Path(base_dir).expanduser()


def _sanitize_pod_title(raw: str) -> str:
    if not raw:
        return ""
    name = os.path.basename(raw).strip()
    name = name.replace(" ", "")
    name = name.replace("/", "").replace("\\", "")
    return name


@router.get("/api/epub/{pod_title}")
async def check_epub(
    pod_title: str,
    session: SessionContainer = Depends(verify_session()),
) -> dict[str, str]:
    folder = _sanitize_pod_title(pod_title)
    if not folder:
        raise HTTPException(status_code=400, detail="Invalid pod title.")
    epub_path = _get_base_dir() / folder / "book.epub"
    if not epub_path.exists():
        raise HTTPException(status_code=404, detail="EPUB file not found.")
    return {"status": "ok", "pod_title": folder}


@router.get("/api/epub_result/{pod_title}")
async def get_epub_result(
    pod_title: str,
    session: SessionContainer = Depends(verify_session()),
) -> Response:
    folder = _sanitize_pod_title(pod_title)
    if not folder:
        raise HTTPException(status_code=400, detail="Invalid pod title.")
    json_path = _get_base_dir() / folder / "book.json"
    if not json_path.exists():
        raise HTTPException(status_code=404, detail="EPUB result not found.")
    try:
        content = json_path.read_text(encoding="utf-8")
    except OSError as exc:
        raise HTTPException(status_code=500, detail="Failed to read EPUB result.") from exc
    return Response(content=content, media_type="application/json")


def _run_epub_parse(pod_title: str) -> None:
    try:
        from agent.epub_agent import run_epub_agent

        run_epub_agent(pod_title)
    except Exception:
        logger.exception("Failed to parse EPUB for %s", pod_title)


def _generate_feed_xml(pod_title: str) -> Path:
    try:
        import feed as feed_module
    except Exception as exc:  # pragma: no cover - import guard
        raise HTTPException(status_code=500, detail="Feed generator is unavailable.") from exc

    base_dir = _get_base_dir()
    book_dir = base_dir / pod_title
    metadata_path = book_dir / "book.json"
    audio_dir = book_dir / "audio"
    output_path = book_dir / "book.xml"

    if not metadata_path.is_file():
        raise HTTPException(status_code=404, detail="Metadata JSON not found.")
    if not audio_dir.is_dir():
        raise HTTPException(status_code=404, detail="Audio directory not found.")

    audio_url_prefix = os.getenv(feed_module.ENV_AUDIO_URL_PREFIX)
    if not audio_url_prefix:
        raise HTTPException(
            status_code=500,
            detail=f"Environment variable {feed_module.ENV_AUDIO_URL_PREFIX} is required.",
        )
    ollama_api_url = os.getenv(feed_module.ENV_OLLAMA_API_URL)
    if not ollama_api_url:
        raise HTTPException(
            status_code=500,
            detail=f"Environment variable {feed_module.ENV_OLLAMA_API_URL} is required.",
        )

    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise HTTPException(status_code=500, detail="Failed to read metadata JSON.") from exc

    media_base_url = f"{audio_url_prefix.rstrip('/')}/{pod_title}"
    audio_base_url = f"{media_base_url}/audio"

    try:
        feed_tree = feed_module.build_feed(
            metadata,
            pod_title,
            audio_dir,
            media_base_url=media_base_url,
            audio_base_url=audio_base_url,
            feed_title=None,
            feed_description=None,
            feed_language=None,
            author_override=None,
            site_url=media_base_url,
            image_url=None,
            explicit=False,
            ollama_api_url=ollama_api_url,
        )
    except (FileNotFoundError, NotADirectoryError) as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Feed generation failed for %s", pod_title)
        raise HTTPException(status_code=500, detail=str(exc) or "Feed generation failed.") from exc

    output_path.parent.mkdir(parents=True, exist_ok=True)
    feed_module._indent(feed_tree.getroot())
    feed_tree.write(output_path, encoding="utf-8", xml_declaration=True)
    return output_path


@router.post("/api/epub_parse/{pod_title}")
async def parse_epub(
    pod_title: str,
    background_tasks: BackgroundTasks,
    session: SessionContainer = Depends(verify_session()),
) -> dict[str, str]:
    folder = _sanitize_pod_title(pod_title)
    if not folder:
        raise HTTPException(status_code=400, detail="Invalid pod title.")
    epub_path = _get_base_dir() / folder / "book.epub"
    if not epub_path.exists():
        raise HTTPException(status_code=404, detail="EPUB file not found.")
    try:
        from agent.epub_agent import run_epub_agent as _unused  # noqa: F401
    except ImportError as exc:
        raise HTTPException(status_code=500, detail="EPUB parser is unavailable.") from exc
    background_tasks.add_task(_run_epub_parse, folder)
    return {"status": "started", "pod_title": folder}


@router.post("/api/feed/{pod_title}")
async def create_feed(
    pod_title: str,
    session: SessionContainer = Depends(verify_session()),
) -> Response:
    folder = _sanitize_pod_title(pod_title)
    if not folder:
        raise HTTPException(status_code=400, detail="Invalid pod title.")
    output_path = _generate_feed_xml(folder)
    try:
        content = output_path.read_text(encoding="utf-8")
    except OSError as exc:
        raise HTTPException(status_code=500, detail="Failed to read feed XML.") from exc
    return Response(content=content, media_type="application/rss+xml")
