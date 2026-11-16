"""Application settings: every path and URL the backend uses, in one place.

settings_from_env() reads them from environment variables; each default below
is the location the ML pipeline writes to (python -m backend.ml.pipeline).

  TOC_DATABASE_URL     SQLite (or other SQLAlchemy) URL      default: backend/toc_workflow.db
  TOC_DATABASE_ECHO    log every SQL statement               default: false
  INIT_DB_ON_STARTUP   create the table if it does not exist default: true
  CORS_ORIGINS         comma-separated allowed origins       default: the Vite dev server
"""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
from typing import List

BACKEND_DIR = Path(__file__).resolve().parent.parent
PROJECT_ROOT = BACKEND_DIR.parent

DEFAULT_DB_FILE = BACKEND_DIR / "toc_workflow.db"
DEFAULT_DATABASE_URL = f"sqlite:///{DEFAULT_DB_FILE.as_posix()}"
DEFAULT_MARKOV_MODEL_PATH = BACKEND_DIR / "ml" / "models" / "markov_chain.json"
DEFAULT_TIMING_MODEL_PATH = BACKEND_DIR / "ml" / "models" / "timing_model.joblib"
DEFAULT_DATASET_PATH = BACKEND_DIR / "ml" / "dataset" / "data" / "orders_event_log.csv"
DEFAULT_RESULTS_PATH = BACKEND_DIR / "ml" / "evaluation" / "results.json"
DEFAULT_FRONTEND_DIR = PROJECT_ROOT / "frontend" / "dist"   # created by `npm run build` in frontend/
DEFAULT_CORS_ORIGINS = ["http://localhost:5173"]            # the Vite dev server (npm run dev)


@dataclass(frozen=True)
class Settings:
    database_url: str
    markov_model_path: Path
    timing_model_path: Path
    dataset_path: Path
    results_path: Path
    frontend_dir: Path
    init_db_on_startup: bool
    cors_origins: List[str]
    database_echo: bool


def _flag(name: str, default: str) -> bool:
    return os.environ.get(name, default).strip().lower() in ("true", "1", "yes")


def settings_from_env() -> Settings:
    origins = [o.strip() for o in os.environ.get("CORS_ORIGINS", ",".join(DEFAULT_CORS_ORIGINS)).split(",") if o.strip()]
    return Settings(
        database_url=os.environ.get("TOC_DATABASE_URL", DEFAULT_DATABASE_URL),
        markov_model_path=DEFAULT_MARKOV_MODEL_PATH,
        timing_model_path=DEFAULT_TIMING_MODEL_PATH,
        dataset_path=DEFAULT_DATASET_PATH,
        results_path=DEFAULT_RESULTS_PATH,
        frontend_dir=DEFAULT_FRONTEND_DIR,
        init_db_on_startup=_flag("INIT_DB_ON_STARTUP", "true"),
        cors_origins=origins,
        database_echo=_flag("TOC_DATABASE_ECHO", "false"),
    )
