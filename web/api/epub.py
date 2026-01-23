from __future__ import annotations

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
