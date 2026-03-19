from __future__ import annotations

import json
import logging
import os
import re
from pathlib import Path
from typing import List

from dotenv import load_dotenv
from fastapi import APIRouter, Depends, HTTPException, Query, Request
from pydantic import BaseModel
from supertokens_python.recipe.session import SessionContainer
from supertokens_python.recipe.session.framework.fastapi import verify_session

logger = logging.getLogger(__name__)
load_dotenv(override=False)

AUDIO_URL_PREFIX_ENV = "AUDIO_URL_PREFIX"
_EPISODE_RE = re.compile(r"^(\d+)_(.+)\.mp3$", re.IGNORECASE)


def _format_duration(seconds: float) -> str:
    total = int(seconds)
    h, rem = divmod(total, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h}:{m:02d}:{s:02d}"
    return f"{m}:{s:02d}"


def _read_duration(path: Path) -> str:
    try:
        from mutagen.mp3 import MP3  # type: ignore
        audio = MP3(str(path))
        return _format_duration(audio.info.length)
    except Exception as exc:
        logger.debug("Could not read duration for %s: %s", path, exc)
        return ""


class EpisodeInfo(BaseModel):
    title: str
    audio_url: str
    pubDate: str = ""
    duration: str = ""
    index: str = ""


class PodInfo(BaseModel):
    id: str
    title: str
    image_url: str
    episodes: List[EpisodeInfo]
    is_owner: bool = False
    visibility: int = 1
    book_id: int = 0
    model_name: str = ""


router = APIRouter()


def _auto_pick_model(book_dir: Path) -> str:
    """Return the first available TTS model directory name under audio/."""
    audio_base = book_dir / "audio"
    if not audio_base.is_dir():
        return ""
    dirs = sorted(d.name for d in audio_base.iterdir() if d.is_dir())
    return dirs[0] if dirs else ""


def _list_episodes(
    book_dir: Path, folder_path: str, model: str, audio_url_prefix: str
) -> List[EpisodeInfo]:
    """Scan audio/{model}/ for {idx}_{title}.mp3 files and return episode list."""
    audio_dir = book_dir / "audio" / model
    if not audio_dir.is_dir():
        return []
    base_url = f"{audio_url_prefix.rstrip('/')}/{folder_path.rstrip('/')}/audio/{model}"
    episodes: List[EpisodeInfo] = []
    for f in sorted(audio_dir.iterdir()):
        if not f.is_file():
            continue
        m = _EPISODE_RE.match(f.name)
        if not m:
            continue
        idx_str, slug = m.group(1), m.group(2)
        title = slug.replace("_", " ").strip()
        episodes.append(
            EpisodeInfo(
                title=title,
                audio_url=f"{base_url}/{f.name}",
                index=str(int(idx_str)),
                duration=_read_duration(f),
            )
        )
    return episodes


def _find_cover_url(book_dir: Path, folder_path: str, audio_url_prefix: str) -> str:
    """Return URL for the first cover image found in the book directory."""
    for name in ("cover.jpg", "cover.jpeg", "cover.png", "cover.webp"):
        if (book_dir / name).exists():
            return f"{audio_url_prefix.rstrip('/')}/{folder_path.rstrip('/')}/{name}"
    return ""


@router.get("/api/get_pods", response_model=List[PodInfo])
async def get_pods(
    request: Request,
    model_name: str = Query(default=""),
    session: SessionContainer = Depends(verify_session()),
) -> List[PodInfo]:
    base_dir = os.getenv("PODS_BASE")
    if not base_dir:
        raise HTTPException(status_code=500, detail="PODS_BASE is not configured")

    root = Path(base_dir).expanduser()
    if not root.exists() or not root.is_dir():
        raise HTTPException(status_code=500, detail="PODS_BASE directory not found")

    audio_url_prefix = os.getenv(AUDIO_URL_PREFIX_ENV, "")
    user_id = session.get_user_id()
    pods: List[PodInfo] = []

    books = request.app.state.pgdb.get_user_books(user_id)
    for book in books:
        folder_path = book["folder_path"]
        book_dir = root / folder_path

        json_path = book_dir / "book.json"
        if not json_path.is_file():
            continue
        try:
            metadata = json.loads(json_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            logger.warning("Skipping book %s: failed to read book.json: %s", folder_path, exc)
            continue

        title = (metadata.get("title") or "").strip() or Path(folder_path).name

        effective_model = model_name.strip() or _auto_pick_model(book_dir)
        if not effective_model:
            continue

        episodes = _list_episodes(book_dir, folder_path, effective_model, audio_url_prefix)
        if not episodes:
            continue

        image_url = _find_cover_url(book_dir, folder_path, audio_url_prefix)
        pod_id = Path(folder_path).name

        pod = PodInfo(id=pod_id, title=title, image_url=image_url, episodes=episodes, model_name=effective_model)
        pod.is_owner = book["user_id"] == user_id
        pod.visibility = book["visibility"]
        pod.book_id = book["id"]
        pods.append(pod)

    return pods
