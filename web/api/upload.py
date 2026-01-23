from __future__ import annotations

import os
import shutil
from pathlib import Path

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
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
    base_dir = _get_base_dir()
    return {"exists": (base_dir / folder).exists(), "folder": folder}


@router.post("/api/upload")
async def upload_book(
    folder_name: str = Form(...),
    file: UploadFile = File(...),
    session: SessionContainer = Depends(verify_session()),
) -> dict[str, str]:
    folder = _sanitize_folder_name(folder_name)
    if not folder:
        raise HTTPException(status_code=400, detail="Invalid folder name.")
    base_dir = _get_base_dir()
    base_dir.mkdir(parents=True, exist_ok=True)
    dest_dir = base_dir / folder
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest_path = dest_dir / "book.epub"
    try:
        with dest_path.open("wb") as target:
            shutil.copyfileobj(file.file, target)
    finally:
        await file.close()
    return {"status": "ok", "folder": folder}
