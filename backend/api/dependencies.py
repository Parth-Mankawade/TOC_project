"""FastAPI dependencies. Everything comes from the app that create_app() built (app.state),
so each app, including each test app, has its own settings, database and models."""

from __future__ import annotations

from typing import Generator

from fastapi import Request
from sqlalchemy.orm import Session

from backend.core.config import Settings
from backend.services.anomaly_service import AnomalyService


def get_settings(request: Request) -> Settings:
    return request.app.state.settings


def get_service(request: Request) -> AnomalyService:
    return request.app.state.service


def get_db_session(request: Request) -> Generator[Session, None, None]:
    session: Session = request.app.state.sessions()
    try:
        yield session
    finally:
        session.close()
