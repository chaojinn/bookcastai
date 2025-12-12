from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from starlette.responses import HTMLResponse, RedirectResponse, Response
from starlette.staticfiles import StaticFiles
import uvicorn

from supertokens_python import InputAppInfo, SupertokensConfig, get_all_cors_headers, init
from supertokens_python.framework.fastapi import get_middleware
from supertokens_python.recipe import emailpassword, session
from supertokens_python.recipe.session.asyncio import get_session
from supertokens_python.recipe.session.exceptions import TryRefreshTokenError

from .PGDB import PGDB
from .api.login import router as login_router
from .api.register import router as register_router
from .api.logoff import router as logoff_router
from .api.get_pods import router as get_pods_router
from .api.last_location import router as last_location_router


def _load_env() -> None:
    """Load environment variables from .env into the current process."""
    load_dotenv(override=False)


def _init_supertokens() -> None:
    """Initialize SuperTokens with configuration drawn from environment variables."""
    _load_env()
    connection_uri = os.getenv("SUPERTOKENS_CORE_URL", "http://localhost:3567")
    api_key = os.getenv("SUPERTOKENS_API_KEY")
    api_domain = os.getenv("API_DOMAIN", "http://localhost:8000")
    website_domain = os.getenv("WEBSITE_DOMAIN", api_domain)

    init(
        app_info=InputAppInfo(
            app_name="ai_pod",
            api_domain=api_domain,
            website_domain=website_domain,
            api_base_path="/auth",
        ),
        framework="fastapi",
        supertokens_config=SupertokensConfig(
            connection_uri=connection_uri,
            api_key=api_key,
        ),
        recipe_list=[
            session.init(
                anti_csrf="NONE",
                cookie_same_site="lax",
                cookie_secure=False,
                get_token_transfer_method=lambda _req, _create, _ctx: "cookie",
            ),
            emailpassword.init(),
        ],
        telemetry=False,
    )


def _build_app() -> FastAPI:
    """Configure the FastAPI application with API routes, static mounts, and 404 handling."""
    _init_supertokens()

    app = FastAPI()
    pgdb = PGDB()
    app.state.pgdb = pgdb

    @app.on_event("startup")
    def _startup() -> None:
        pgdb.connect()
        pgdb.init()

    @app.on_event("shutdown")
    def _shutdown() -> None:
        pgdb.disconnect()

    app.add_middleware(get_middleware())

    cors_origins = [os.getenv("WEBSITE_DOMAIN", "http://localhost:8000")]
    app.add_middleware(
        CORSMiddleware,
        allow_origins=cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"] + get_all_cors_headers(),
    )

    app.include_router(register_router)
    app.include_router(login_router)
    app.include_router(logoff_router)
    app.include_router(get_pods_router)
    app.include_router(last_location_router)

    base_dir = Path(__file__).resolve().parent
    html_dir = base_dir / "html"
    css_dir = base_dir / "css"
    js_dir = base_dir / "js"
    pods_dir = Path(os.getenv("PODS_BASE", "")).expanduser()

    not_found_body = (
        "<!doctype html><html><head><title>404</title></head>"
        "<body><h1>Page not found</h1></body></html>"
    )

    @app.middleware("http")
    async def _transform_404(request, call_next):  # type: ignore[unused-arg]
        response = await call_next(request)
        if response.status_code == 404:
            return HTMLResponse(not_found_body, status_code=404)
        return response

    @app.middleware("http")
    async def _enforce_api_auth(request, call_next):  # type: ignore[unused-arg]
        path = request.url.path
        if path.startswith("/api") and path not in {"/api/login", "/api/register"}:
            try:
                session_container = await get_session(
                    request,
                    session_required=True,
                    anti_csrf_check=False,
                )
            except TryRefreshTokenError:
                return Response(status_code=401)
            if session_container is None:
                return Response(status_code=401)
        return await call_next(request)

    @app.middleware("http")
    async def _enforce_login_redirect(request, call_next):  # type: ignore[unused-arg]
        path = request.url.path
        if path.startswith("/api") or path.startswith("/css") or path.startswith("/js"):
            return await call_next(request)

        allowed_html = {"/login.html", "/register.html"}
        if path in allowed_html:
            return await call_next(request)

        if path == "/":
            return RedirectResponse("/login.html")

        if path.endswith(".html"):
            try:
                session_container = await get_session(
                    request,
                    session_required=False,
                    anti_csrf_check=False,
                )
            except TryRefreshTokenError:
                return RedirectResponse("/login.html")
            if session_container is not None:
                return await call_next(request)
            return RedirectResponse("/login.html")

        return await call_next(request)

    @app.middleware("http")
    async def _record_last_location(request, call_next):  # type: ignore[unused-arg]
        response = await call_next(request)
        try:
            if (
                request.method == "GET"
                and request.url.path.endswith(".html")
                and not request.url.path.endswith("/index.html")
                and response.status_code < 400
            ):
                session_container = await get_session(
                    request,
                    session_required=False,
                    anti_csrf_check=False,
                )
                if session_container is not None:
                    try:
                        uri = request.url.path
                        if request.url.query:
                            uri = f"{uri}?{request.url.query}"
                        request.app.state.pgdb.update_last_location(session_container.get_user_id(), uri)
                    except Exception:
                        # Best-effort: do not block the response if logging fails.
                        pass
        except TryRefreshTokenError:
            pass
        return response

    app.mount("/css", StaticFiles(directory=css_dir), name="css")
    app.mount("/js", StaticFiles(directory=js_dir), name="js")
    if pods_dir.exists():
        app.mount("/pods", StaticFiles(directory=pods_dir), name="pods")
    app.mount("/", StaticFiles(directory=html_dir, html=True), name="html")

    return app


app = _build_app()


def main() -> None:
    """Run the uvicorn server."""
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "8000"))
    uvicorn.run("web.server:app", host=host, port=port, reload=False)


if __name__ == "__main__":
    main()
