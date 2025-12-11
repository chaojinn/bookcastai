from __future__ import annotations

import os
from typing import Any, Dict

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, EmailStr
from supertokens_python.recipe.emailpassword.asyncio import sign_in
from supertokens_python.recipe.emailpassword.interfaces import WrongCredentialsError
from supertokens_python.recipe.session.asyncio import create_new_session


class LoginRequest(BaseModel):
    email: EmailStr
    password: str


router = APIRouter()


@router.post("/api/login")
async def login(request: Request, payload: LoginRequest) -> Dict[str, Any]:
    tenant_id = os.getenv("SUPERTOKENS_TENANT_ID", "public")
    result = await sign_in(tenant_id, payload.email, payload.password)
    if isinstance(result, WrongCredentialsError):
        raise HTTPException(status_code=401, detail="Invalid credentials")
    await create_new_session(
        request,
        tenant_id,
        result.recipe_user_id,
    )
    return {
        "user_id": result.user.id,
        "recipe_user_id": result.recipe_user_id.get_as_string(),
        "email": result.user.emails[0] if result.user.emails else payload.email,
    }
