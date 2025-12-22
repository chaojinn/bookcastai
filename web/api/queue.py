from __future__ import annotations

from typing import List, Literal

from fastapi import APIRouter, Depends, Request
from pydantic import BaseModel, Field
from supertokens_python.recipe.session import SessionContainer
from supertokens_python.recipe.session.framework.fastapi import verify_session

router = APIRouter()


class QueueItemPayload(BaseModel):
    pod_id: str = Field(..., min_length=1)
    episode_idx: int
    start_pos: int = 0
    is_current: bool = False


class QueueItem(BaseModel):
    idx: int
    pod_id: str
    episode_idx: int
    start_pos: int
    is_current: bool


class MovePayload(BaseModel):
    idx: int
    direction: Literal["up", "down"]


@router.post("/api/queue")
async def add_to_queue(
    payload: QueueItemPayload,
    request: Request,
    session: SessionContainer = Depends(verify_session()),
) -> dict[str, bool]:
    """Add an episode to the user's queue; returns added=False if duplicate exists."""
    added = request.app.state.pgdb.add_to_queue(
        session.get_user_id(),
        payload.pod_id,
        payload.episode_idx,
        payload.start_pos,
        payload.is_current,
    )
    return {"added": added}


@router.get("/api/queue", response_model=List[QueueItem])
async def get_queue(
    request: Request,
    session: SessionContainer = Depends(verify_session()),
) -> List[QueueItem]:
    """Return the user's queue ordered by idx."""
    items = request.app.state.pgdb.get_player_queue(session.get_user_id())
    return [QueueItem(**item) for item in items]


@router.post("/api/queue/move")
async def move_in_queue(
    payload: MovePayload,
    request: Request,
    session: SessionContainer = Depends(verify_session()),
) -> dict[str, bool]:
    """Move a queue item up or down."""
    moved = request.app.state.pgdb.move_queue_item(
        session.get_user_id(), payload.idx, payload.direction
    )
    return {"moved": moved}


@router.delete("/api/queue/{idx}")
async def delete_from_queue(
    idx: int,
    request: Request,
    session: SessionContainer = Depends(verify_session()),
) -> dict[str, bool]:
    """Delete a queue item and compact indices."""
    removed = request.app.state.pgdb.delete_queue_item(session.get_user_id(), idx)
    return {"removed": removed}
