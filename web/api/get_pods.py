from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import List, Optional
import xml.etree.ElementTree as ET

from dotenv import load_dotenv
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

logger = logging.getLogger(__name__)
load_dotenv(override=False)


class EpisodeInfo(BaseModel):
    title: str
    audio_url: str


class PodInfo(BaseModel):
    title: str
    image_url: str
    episodes: List[EpisodeInfo]


router = APIRouter()


def _parse_pod_file(path: Path) -> Optional[PodInfo]:
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
        enclosure = item.find("enclosure")
        audio_url = ""
        if enclosure is not None:
            audio_url = (enclosure.attrib.get("url") or "").strip()
        if not audio_url:
            audio_url = (item.findtext("link") or "").strip()
        if not ep_title and not audio_url:
            continue
        episodes.append(EpisodeInfo(title=ep_title, audio_url=audio_url))

    if not title and not image_url and not episodes:
        return None

    return PodInfo(title=title, image_url=image_url, episodes=episodes)


@router.get("/api/get_pods", response_model=List[PodInfo])
async def get_pods() -> List[PodInfo]:
    base_dir = os.getenv("PODS_BASE")
    if not base_dir:
        raise HTTPException(status_code=500, detail="PODS_BASE is not configured")

    root = Path(base_dir).expanduser()
    if not root.exists() or not root.is_dir():
        raise HTTPException(status_code=500, detail="PODS_BASE directory not found")

    pods: List[PodInfo] = []
    for xml_path in sorted(root.rglob("*.xml")):
        pod = _parse_pod_file(xml_path)
        if pod:
            pods.append(pod)

    return pods
