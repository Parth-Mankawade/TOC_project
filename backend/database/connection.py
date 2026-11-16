"""Database engine, sessions and table setup (STORAGE checkpoint).

The app builds its own engine from Settings (backend/main.py), so tests can use a
temporary database without changing any global state.
"""

from __future__ import annotations

from sqlalchemy import Engine, create_engine, inspect
from sqlalchemy.orm import Session, sessionmaker

from backend.core.errors import StorageError
from backend.database.models import Base, WorkflowAnalysis


def make_engine(database_url: str, echo: bool) -> Engine:
    connect_args = {"check_same_thread": False} if database_url.startswith("sqlite") else {}
    return create_engine(database_url, echo=echo, connect_args=connect_args)


def make_session_factory(engine: Engine) -> sessionmaker[Session]:
    return sessionmaker(bind=engine, autocommit=False, autoflush=False, expire_on_commit=False)


def check_table(engine: Engine) -> None:
    """Raise StorageError unless the analysis table exists with every column the code uses."""
    table = WorkflowAnalysis.__tablename__
    inspector = inspect(engine)
    if not inspector.has_table(table):
        raise StorageError(f"a '{table}' table (created on startup when INIT_DB_ON_STARTUP is true)",
                           "no such table", engine.url.render_as_string(hide_password=True))
    present = {c["name"] for c in inspector.get_columns(table)}
    missing = sorted({c.name for c in WorkflowAnalysis.__table__.columns} - present)
    if missing:
        raise StorageError(f"every column of '{table}' (an older database file? delete it to recreate)",
                           f"missing {missing}", engine.url.render_as_string(hide_password=True))


def init_db(engine: Engine) -> None:
    """Create the table if it does not exist, then check its columns (create_all never alters a table)."""
    Base.metadata.create_all(bind=engine)
    check_table(engine)
