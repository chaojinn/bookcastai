from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, Depends, Request
from supertokens_python.recipe.session import SessionContainer
from supertokens_python.recipe.session.framework.fastapi import verify_session

router = APIRouter()


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
