"""Run every test in the project with one command:

    .venv\\Scripts\\python.exe run_tests.py

1. Backend: pytest on backend/tests (data generator, models, pipeline, API, end-to-end).
2. Frontend: TypeScript type check, then the Vitest unit tests (needs `npm install` in frontend/ once).

Stops at the first failing step and exits with its exit code.
"""

from __future__ import annotations

from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
FRONTEND = ROOT / "frontend"


def npm() -> str:
    found = shutil.which("npm")
    if found is None:
        sys.exit("[TESTS] expected npm on PATH (install Node.js 20+), got none; input=run_tests.py")
    return found


STEPS = [
    ("backend tests", [sys.executable, "-m", "pytest", "backend/tests", "-q", "-p", "no:cacheprovider"], ROOT),
    ("frontend type check", [npm(), "run", "typecheck"], FRONTEND),
    ("frontend tests", [npm(), "test"], FRONTEND),
]

if __name__ == "__main__":
    if not (FRONTEND / "node_modules").is_dir():
        sys.exit("[TESTS] expected frontend/node_modules, got none; run `npm install` in frontend/ first")
    for name, command, cwd in STEPS:
        print(f"\n=== {name} ===", flush=True)
        code = subprocess.call(command, cwd=cwd)
        if code != 0:
            print(f"\n[TESTS] {name} failed with exit code {code}")
            sys.exit(code)
    print("\nAll test steps passed.")
