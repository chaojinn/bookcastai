from __future__ import annotations

from fastapi import APIRouter, Depends
from supertokens_python.recipe.session import SessionContainer
from supertokens_python.recipe.session.framework.fastapi import verify_session

router = APIRouter()


@router.post("/api/logoff")
async def logoff(session: SessionContainer = Depends(verify_session())) -> dict[str, str]:
    await session.revoke_session()
    return {"status": "OK"}
