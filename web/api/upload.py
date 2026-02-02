from __future__ import annotations

import os
import shutil
from pathlib import Path

from fastapi import APIRouter, Depends, File, Form, HTTPException, Request, UploadFile
from supertokens_python.recipe.session import SessionContainer
from supertokens_python.recipe.session.framework.fastapi import verify_session

router = APIRouter()


def _get_base_dir() -> Path:
    base_dir = os.getenv("PODS_BASE")
    if not base_dir:
        raise HTTPException(status_code=500, detail="PODS_BASE is not configured.")
    return Path(base_dir).expanduser()


def _sanitize_folder_name(raw: str) -> str:
    if not raw:
        return ""
    name = os.path.basename(raw).strip()
    name = name.replace(" ", "")
    name = name.replace("/", "").replace("\\", "")
    return name


@router.get("/api/upload/check")
async def check_upload_folder(
    name: str,
    session: SessionContainer = Depends(verify_session()),
) -> dict[str, object]:
    folder = _sanitize_folder_name(name)
    if not folder:
        raise HTTPException(status_code=400, detail="Invalid folder name.")
    user_id = session.get_user_id()
    base_dir = _get_base_dir()
    user_folder = base_dir / user_id / folder
    return {"exists": user_folder.exists(), "folder": folder}


@router.post("/api/upload")
async def upload_book(
    request: Request,
    folder_name: str = Form(...),
    file: UploadFile = File(...),
    visibility: int = Form(1),
    session: SessionContainer = Depends(verify_session()),
) -> dict[str, str]:
    folder = _sanitize_folder_name(folder_name)
    if not folder:
        raise HTTPException(status_code=400, detail="Invalid folder name.")
    user_id = session.get_user_id()
    base_dir = _get_base_dir()
    # User-specific directory structure
    user_dir = base_dir / user_id
    user_dir.mkdir(parents=True, exist_ok=True)
    dest_dir = user_dir / folder
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest_path = dest_dir / "book.epub"
    try:
        with dest_path.open("wb") as target:
            shutil.copyfileobj(file.file, target)
    finally:
        await file.close()
    # Insert/update database record
    folder_path = f"{user_id}/{folder}"
    visibility_val = 1 if visibility else 0
    request.app.state.pgdb.insert_book(user_id, folder, folder_path, visibility_val)
    return {"status": "ok", "folder": folder, "path": folder_path}
