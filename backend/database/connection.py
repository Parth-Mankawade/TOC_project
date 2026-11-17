"""Database engine, sessions and table setup (STORAGE checkpoint).

The app builds its own engine from Settings (backend/main.py), so tests can use a
temporary database without changing any global state.
"""

from __future__ import annotations

from sqlalchemy import Engine, create_engine, inspect, select
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
    check_rows_current(engine)


# Timing results stored before the timing model had three checks lack these keys. They cannot be
# shown with the current schema, and their scores used another threshold, so they are not converted.
CURRENT_TIMING_KEYS = ("checks", "trend_z")


def check_rows_current(engine: Engine) -> None:
    """Raise StorageError if any stored timing result comes from an older model version."""
    with engine.connect() as conn:
        rows = conn.execute(select(WorkflowAnalysis.id, WorkflowAnalysis.timing_evaluation)
                            .where(WorkflowAnalysis.timing_evaluation.is_not(None))).all()
    old = [row.id for row in rows
           if isinstance(row.timing_evaluation, dict) and not all(k in row.timing_evaluation for k in CURRENT_TIMING_KEYS)]
    if old:
        raise StorageError(
            "analyses stored by the current model version (an older database file: stop the server, delete it "
            "and start again; it is recreated empty)",
            f"{len(old)} record(s) with timing results from an older version, e.g. id {old[0]}",
            engine.url.render_as_string(hide_password=True))


def init_db(engine: Engine) -> None:
    """Create the table if it does not exist, then check its columns (create_all never alters a table)."""
    Base.metadata.create_all(bind=engine)
    check_table(engine)
