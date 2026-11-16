"""FastAPI application: create_app(settings) builds one app with its own database engine,
sessions and analysis service, so tests can run a real app on temporary files.

Every error leaves as the same JSON shape with the real message:
    {"error": {"code": ..., "checkpoint": ..., "message": "[CHECKPOINT] expected ..., got ...; input=...",
               "details": [...] or null}}

  InputError (bad request)                          422
  NotFoundError (no such record)                    404
  ModelLoadError, DataLoadError, EvaluateError,
  StorageError (a file or the database is missing)  503
  any other pipeline error, or an unexpected error  500 (message kept, never replaced)

Run with:  .venv\\Scripts\\python.exe -m uvicorn backend.main:app --port 8000
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from http import HTTPStatus
import re
from typing import Any, List, Optional

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from starlette.exceptions import HTTPException as StarletteHTTPException

from backend.api.routes import root_router, router
from backend.core.config import Settings, settings_from_env
from backend.core.errors import (
    DataLoadError,
    EvaluateError,
    InputError,
    ModelLoadError,
    NotFoundError,
    PipelineError,
    StorageError,
)
from backend.database.connection import init_db, make_engine, make_session_factory
from backend.services.anomaly_service import AnomalyService

ERROR_STATUS = [
    (InputError, 422),
    (NotFoundError, 404),
    ((ModelLoadError, DataLoadError, EvaluateError, StorageError), 503),
]
SERVER_ERROR = 500  # a pipeline step itself failed (FEATURES, ANALYZE, ...): a bug, not the caller's fault


def _body(code: str, checkpoint: str, message: str, details: Optional[List[str]] = None) -> dict:
    return {"error": {"code": code, "checkpoint": checkpoint, "message": message, "details": details}}


def _code(exc: Exception) -> str:
    """InputError -> INPUT_ERROR, ModelLoadError -> MODEL_LOAD_ERROR."""
    return re.sub(r"(?<!^)(?=[A-Z])", "_", type(exc).__name__).upper()


def pipeline_error_response(exc: PipelineError) -> JSONResponse:
    status = next((code for kinds, code in ERROR_STATUS if isinstance(exc, kinds)), SERVER_ERROR)
    return JSONResponse(status_code=status, content=_body(_code(exc), exc.checkpoint, str(exc)))


def validation_error_response(exc: RequestValidationError) -> JSONResponse:
    """422 with one '[INPUT] expected ..., got ...; input=field' line per problem."""
    lines: List[str] = []
    for err in exc.errors():
        cause: Any = err["ctx"]["error"] if "ctx" in err and "error" in err["ctx"] else None
        if isinstance(cause, PipelineError):
            lines.append(str(cause))
        else:
            where = ".".join(str(p) for p in err["loc"])
            lines.append(f"[INPUT] expected a valid value ({err['msg']}), got {err['input']!r}; input={where}")
    message = lines[0] if len(lines) == 1 else f"{lines[0]} (and {len(lines) - 1} more)"
    return JSONResponse(status_code=422, content=_body("INPUT_ERROR", "INPUT", message, lines))


def http_error_response(request: Request, exc: StarletteHTTPException) -> JSONResponse:
    status = HTTPStatus(exc.status_code)
    expected = {404: "an existing endpoint", 405: "a method this endpoint accepts"}
    message = (f"[API] expected {expected[status.value] if status.value in expected else 'a successful request'}, "
               f"got {status.value} {status.phrase}; input={request.method} {request.url.path}")
    return JSONResponse(status_code=status.value, content=_body(status.name, "API", message), headers=exc.headers)


def unexpected_error_response(exc: Exception, where: str = "request") -> JSONResponse:
    message = f"[API] expected the request to complete, got {type(exc).__name__}: {exc}; input={where}"
    return JSONResponse(status_code=500, content=_body("INTERNAL_ERROR", "API", message))


def create_app(settings: Optional[Settings] = None) -> FastAPI:
    settings = settings_from_env() if settings is None else settings
    engine = make_engine(settings.database_url, settings.database_echo)

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        if settings.init_db_on_startup:
            init_db(engine)
        yield
        engine.dispose()

    app = FastAPI(title="Workflow Anomaly Detection API", version="2.0.0", lifespan=lifespan,
                  description="Finite automaton rules plus ML (Markov chain and a timing/amount model) "
                              "for online order workflows.")
    app.state.settings = settings
    app.state.engine = engine
    app.state.sessions = make_session_factory(engine)
    app.state.service = AnomalyService(settings.markov_model_path, settings.timing_model_path)

    app.add_middleware(CORSMiddleware, allow_origins=settings.cors_origins, allow_credentials=True,
                       allow_methods=["GET", "POST"], allow_headers=["Content-Type"])

    @app.exception_handler(PipelineError)
    async def _pipeline(_: Request, exc: PipelineError) -> JSONResponse:
        return pipeline_error_response(exc)

    @app.exception_handler(RequestValidationError)
    async def _validation(_: Request, exc: RequestValidationError) -> JSONResponse:
        return validation_error_response(exc)

    @app.exception_handler(StarletteHTTPException)
    async def _http(request: Request, exc: StarletteHTTPException) -> JSONResponse:
        return http_error_response(request, exc)

    @app.exception_handler(Exception)
    async def _unexpected(request: Request, exc: Exception) -> JSONResponse:
        return unexpected_error_response(exc, f"{request.method} {request.url.path}")

    app.include_router(root_router)
    app.include_router(router)

    # The built dashboard (npm run build in frontend/) is served at /app/. Without a build,
    # "/" is a normal 404 so a missing build is visible rather than an empty page.
    if settings.frontend_dir.is_dir():
        app.mount("/app", StaticFiles(directory=settings.frontend_dir, html=True), name="dashboard")

        @app.get("/", include_in_schema=False)
        def open_dashboard() -> RedirectResponse:
            return RedirectResponse(url="/app/")

    return app


app = create_app()
