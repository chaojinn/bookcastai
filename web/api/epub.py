from __future__ import annotations

import json
import logging
import os
from pathlib import Path

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Request
from fastapi.responses import Response
from pydantic import BaseModel
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


def _get_book_dir(request: Request, user_id: str, pod_title: str) -> Path:
    """Get the book directory, checking database first then falling back to legacy path."""
    base_dir = _get_base_dir()
    folder = _sanitize_pod_title(pod_title)
    if not folder:
        raise HTTPException(status_code=400, detail="Invalid pod title.")
    # Check database for the book
    book = request.app.state.pgdb.get_book_for_user(folder, user_id)
    if book:
        return base_dir / book["folder_path"]
    # Fallback to legacy flat path
    legacy_path = base_dir / folder
    if legacy_path.exists():
        return legacy_path
    raise HTTPException(status_code=404, detail="Book not found.")


@router.get("/api/epub/{pod_title}")
async def check_epub(
    pod_title: str,
    request: Request,
    session: SessionContainer = Depends(verify_session()),
) -> dict[str, str]:
    user_id = session.get_user_id()
    book_dir = _get_book_dir(request, user_id, pod_title)
    epub_path = book_dir / "book.epub"
    if not epub_path.exists():
        raise HTTPException(status_code=404, detail="EPUB file not found.")
    return {"status": "ok", "pod_title": _sanitize_pod_title(pod_title)}


@router.get("/api/epub_result/{pod_title}")
async def get_epub_result(
    pod_title: str,
    request: Request,
    session: SessionContainer = Depends(verify_session()),
) -> Response:
    user_id = session.get_user_id()
    book_dir = _get_book_dir(request, user_id, pod_title)
    json_path = book_dir / "book.json"
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


def _generate_feed_xml(book_dir: Path, pod_title: str, folder_path: str, model_name: str = "kokoro") -> Path:
    try:
        import feed as feed_module
    except Exception as exc:  # pragma: no cover - import guard
        raise HTTPException(status_code=500, detail="Feed generator is unavailable.") from exc

    safe_model = model_name.strip().lower().replace("/", "").replace("\\", "").removesuffix("-tts") or "kokoro"
    metadata_path = book_dir / "book.json"
    audio_dir = book_dir / "audio" / safe_model
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

    # Use folder_path for URL generation (includes user_id for new structure)
    media_base_url = f"{audio_url_prefix.rstrip('/')}/{folder_path}"
    audio_base_url = f"{media_base_url}/audio/{safe_model}"

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


class SaveJsonRequest(BaseModel):
    pod_title: str
    content: str


@router.post("/api/epub/json")
async def save_epub_json(
    body: SaveJsonRequest,
    request: Request,
    session: SessionContainer = Depends(verify_session()),
) -> dict[str, str]:
    user_id = session.get_user_id()
    book_dir = _get_book_dir(request, user_id, body.pod_title)
    try:
        json.loads(body.content)
    except (json.JSONDecodeError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=f"Invalid JSON: {exc}") from exc
    json_path = book_dir / "book.json"
    try:
        json_path.write_text(body.content, encoding="utf-8")
    except OSError as exc:
        raise HTTPException(status_code=500, detail="Failed to write JSON file.") from exc
    return {"status": "ok"}


@router.post("/api/epub_parse/{pod_title}")
async def parse_epub(
    pod_title: str,
    request: Request,
    background_tasks: BackgroundTasks,
    session: SessionContainer = Depends(verify_session()),
) -> dict[str, str]:
    user_id = session.get_user_id()
    folder = _sanitize_pod_title(pod_title)
    if not folder:
        raise HTTPException(status_code=400, detail="Invalid pod title.")
    # Get book directory (user-specific or legacy)
    book = request.app.state.pgdb.get_book_for_user(folder, user_id)
    if book:
        folder_path = book["folder_path"]
    else:
        folder_path = folder  # Legacy flat structure
    book_dir = _get_base_dir() / folder_path
    epub_path = book_dir / "book.epub"
    if not epub_path.exists():
        raise HTTPException(status_code=404, detail="EPUB file not found.")
    try:
        from agent.epub_agent import run_epub_agent as _unused  # noqa: F401
    except ImportError as exc:
        raise HTTPException(status_code=500, detail="EPUB parser is unavailable.") from exc
    background_tasks.add_task(_run_epub_parse, folder_path)
    return {"status": "started", "pod_title": folder}


@router.get("/api/audio_models/{pod_title}")
async def list_audio_models(
    pod_title: str,
    request: Request,
    session: SessionContainer = Depends(verify_session()),
) -> list[str]:
    user_id = session.get_user_id()
    book_dir = _get_book_dir(request, user_id, pod_title)
    audio_dir = book_dir / "audio"
    if not audio_dir.is_dir():
        return []
    return sorted(d.name for d in audio_dir.iterdir() if d.is_dir())


@router.post("/api/feed/{pod_title}")
async def create_feed(
    pod_title: str,
    request: Request,
    model_name: str = "kokoro",
    session: SessionContainer = Depends(verify_session()),
) -> Response:
    user_id = session.get_user_id()
    folder = _sanitize_pod_title(pod_title)
    if not folder:
        raise HTTPException(status_code=400, detail="Invalid pod title.")
    book_dir = _get_book_dir(request, user_id, folder)
    folder_path = str(book_dir.relative_to(_get_base_dir()))
    output_path = _generate_feed_xml(book_dir, folder, folder_path, model_name)
    try:
        content = output_path.read_text(encoding="utf-8")
    except OSError as exc:
        raise HTTPException(status_code=500, detail="Failed to read feed XML.") from exc
    return Response(content=content, media_type="application/rss+xml")
