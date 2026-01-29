from __future__ import annotations

import logging
from typing import Dict, List

from fastapi import APIRouter, Depends, HTTPException, Query
from supertokens_python.recipe.session import SessionContainer
from supertokens_python.recipe.session.framework.fastapi import verify_session

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get("/api/tts/voices")
async def get_tts_voices(
    model_name: str | None = Query(default=None),
    session: SessionContainer = Depends(verify_session()),
) -> List[Dict[str, str]]:
    if not model_name:
        return []

    normalized = model_name.strip().lower()
    if normalized in {"kokoro-tts", "kokoro"}:
        try:
            from tts.kokoro import get_english_voices
        except ImportError as exc:
            logger.exception("Failed to load Kokoro voices")
            raise HTTPException(status_code=500, detail="Kokoro voices unavailable.") from exc
        return get_english_voices()

    return []
