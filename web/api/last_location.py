from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, Depends, Request
from pydantic import BaseModel
from supertokens_python.recipe.session import SessionContainer
from supertokens_python.recipe.session.framework.fastapi import verify_session

router = APIRouter()


class LastLocationPayload(BaseModel):
    uri: str


@router.get("/api/last_location")
async def get_last_location(
    request: Request,
    session: SessionContainer = Depends(verify_session()),
) -> dict[str, Optional[str]]:
    """
    Return the last recorded URI for the authenticated user.
    """
    uri = request.app.state.pgdb.get_last_location(session.get_user_id())
    return {"uri": uri}


@router.post("/api/last_location")
async def set_last_location(
    payload: LastLocationPayload,
    request: Request,
    session: SessionContainer = Depends(verify_session()),
) -> dict[str, str]:
    """Store the provided uri as the user's last location."""
    request.app.state.pgdb.update_last_location(session.get_user_id(), payload.uri)
    return {"status": "ok"}
