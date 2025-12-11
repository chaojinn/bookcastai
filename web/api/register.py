from __future__ import annotations

import os
from typing import Any, Dict

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, EmailStr
from supertokens_python.recipe.emailpassword.asyncio import sign_up
from supertokens_python.recipe.emailpassword.interfaces import EmailAlreadyExistsError


class RegisterRequest(BaseModel):
    email: EmailStr
    password: str


router = APIRouter()


@router.post("/api/register")
async def register(payload: RegisterRequest) -> Dict[str, Any]:
    tenant_id = os.getenv("SUPERTOKENS_TENANT_ID", "public")
    result = await sign_up(tenant_id, payload.email, payload.password)
    if isinstance(result, EmailAlreadyExistsError):
        raise HTTPException(status_code=409, detail="User already exists")
    return {
        "user_id": result.user.id,
        "recipe_user_id": result.recipe_user_id.get_as_string(),
        "email": result.user.emails[0] if result.user.emails else payload.email,
    }
