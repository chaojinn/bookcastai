from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import List, Optional
import xml.etree.ElementTree as ET

from dotenv import load_dotenv
from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel
from supertokens_python.recipe.session import SessionContainer
from supertokens_python.recipe.session.framework.fastapi import verify_session
logger = logging.getLogger(__name__)
load_dotenv(override=False)


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


router = APIRouter()


def _parse_pod_file(path: Path) -> Optional[PodInfo]:
    pod_id = path.parent.name
    try:
        root = ET.parse(path).getroot()
    except ET.ParseError as exc:
        logger.warning("Skipping invalid XML at %s: %s", path, exc)
        return None

    channel = root.find("channel")
    if channel is None:
        logger.debug("No channel element found in %s", path)
        return None

    title = (channel.findtext("title") or "").strip()
    image_url = (channel.findtext("image/url") or "").strip()
    episodes: List[EpisodeInfo] = []

    for item in channel.findall("item"):
        ep_title = (item.findtext("title") or "").strip()
        pub_date = (item.findtext("pubDate") or "").strip()
        duration = (
            item.findtext("itunes:duration")
            or item.findtext("{http://www.itunes.com/dtds/podcast-1.0.dtd}duration")
            or ""
        ).strip()
        index = (
            item.findtext("itunes:episode")
            or item.findtext("{http://www.itunes.com/dtds/podcast-1.0.dtd}episode")
            or ""
        ).strip()
        enclosure = item.find("enclosure")
        audio_url = ""
        if enclosure is not None:
            audio_url = (enclosure.attrib.get("url") or "").strip()
        if not audio_url:
            audio_url = (item.findtext("link") or "").strip()
        if not ep_title and not audio_url:
            continue
        episodes.append(
            EpisodeInfo(
                title=ep_title,
                audio_url=audio_url,
                pubDate=pub_date,
                duration=duration,
                index=index,
            )
        )

    if not title and not image_url and not episodes:
        return None

    return PodInfo(id=pod_id, title=title, image_url=image_url, episodes=episodes)


@router.get("/api/get_pods", response_model=List[PodInfo])
async def get_pods(
    request: Request,
    session: SessionContainer = Depends(verify_session()),
) -> List[PodInfo]:
    base_dir = os.getenv("PODS_BASE")
    if not base_dir:
        raise HTTPException(status_code=500, detail="PODS_BASE is not configured")

    root = Path(base_dir).expanduser()
    if not root.exists() or not root.is_dir():
        raise HTTPException(status_code=500, detail="PODS_BASE directory not found")

    user_id = session.get_user_id()
    pods: List[PodInfo] = []
    seen_ids: set[str] = set()

    # Get accessible books from database (user's own + public)
    books = request.app.state.pgdb.get_user_books(user_id)
    for book in books:
        folder_path = book["folder_path"]
        xml_path = root / folder_path / "book.xml"
        if not xml_path.exists():
            continue
        pod = _parse_pod_file(xml_path)
        if pod:
            pod.is_owner = (book["user_id"] == user_id)
            pod.visibility = book["visibility"]
            pods.append(pod)
            seen_ids.add(pod.id)

   

    return pods
