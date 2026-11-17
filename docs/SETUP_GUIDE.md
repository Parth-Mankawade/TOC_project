# Setup Guide: running the project on a new computer

This guide takes a computer with nothing installed to the running dashboard. It takes about
10 minutes, most of it downloads. Commands are for **Windows PowerShell**; macOS/Linux versions
are given where they differ.

---

## 1. Install the two tools (once per computer)

| Tool | Version | Get it from | Check it worked |
| :--- | :--- | :--- | :--- |
| **Python** | 3.13 | https://www.python.org/downloads/ (on Windows tick **"Add python.exe to PATH"**) | `py -3.13 --version` (macOS/Linux: `python3.13 --version`) |
| **Node.js** | 20 or newer (LTS) | https://nodejs.org/ | `node --version` and `npm --version` |

Git is only needed if you clone the project; copying the folder works too.

## 2. Get the project folder

```powershell
git clone https://github.com/Parth-Mankawade/TOC_project.git TOC
cd TOC
```

or copy the `TOC` folder onto the computer and open PowerShell inside it (you can skip `.venv`,
`frontend/node_modules` and `frontend/dist`, which steps 3 and 4 recreate).

**Every command below is run from this `TOC` folder** unless it says `cd frontend`.

## 3. Set up Python (once)

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r backend\requirements.txt
```

macOS/Linux:

```bash
python3.13 -m venv .venv
.venv/bin/python -m pip install -r backend/requirements.txt
```

This creates a private Python environment in `.venv` and installs FastAPI, scikit-learn 1.9.1, SciPy,
SQLAlchemy, pytest and the rest. (macOS/Linux: in every later command, write `.venv/bin/python`
instead of `.\.venv\Scripts\python.exe`.)

## 4. Set up and build the dashboard (once, and again after changing frontend code)

```powershell
cd frontend
npm install
npm run build
cd ..
```

`npm run build` creates `frontend/dist/`, which the backend serves as the dashboard.

> If PowerShell says *"running scripts is disabled on this system"*, use `npm.cmd install` and
> `npm.cmd run build` instead.

## 5. Generate the data and train the models (once; about half a minute)

```powershell
.\.venv\Scripts\python.exe -m backend.ml.pipeline
```

This one command:

1. generates the synthetic event log (12,350 orders) → `backend/ml/dataset/data/orders_event_log.csv`
2. trains the Markov chain → `backend/ml/models/markov_chain.json`
3. trains three timing-model candidates, picks the best on validation data → `backend/ml/models/timing_model.joblib` (+ `.json`)
4. sets one threshold for all four checks on validation data (stored in both model files)
5. evaluates every model on the test data → `backend/ml/evaluation/results.json`

It ends by printing the results table. The same seed always gives the same data and results, so
the numbers match the documentation on every computer.

## 6. Start the app

```powershell
.\.venv\Scripts\python.exe -m uvicorn backend.main:app --port 8000
```

Leave this window open (the app runs while it is open; **Ctrl+C** stops it). Then open:

- **http://127.0.0.1:8000/** : the dashboard (redirects to `/app/`)
- **http://127.0.0.1:8000/docs** : interactive API documentation

The bottom-left of the dashboard should say **Backend ready · Database ok · Markov chain + k-means
clustering loaded**. The database starts empty: press **Start stream** on the Overview page to fill it.

## 7. Run the tests (optional, shows everything works)

```powershell
.\.venv\Scripts\python.exe run_tests.py
```

Expected end of the output: a line ending in `passed` for the backend and for the frontend (no
`failed`), and **All test steps passed.** (about one and a half minutes).

---

## Every day after setup

Only step 6 is needed:

```powershell
.\.venv\Scripts\python.exe -m uvicorn backend.main:app --port 8000
```

Re-run step 5 only if you change the data generator or the models; re-run step 4 only if you change
frontend code.

## Working on the frontend with live reload (optional)

Keep the backend (step 6) running, then in a second window:

```powershell
cd frontend
npm run dev
```

and open http://localhost:5173/app/. Changes to the React code appear immediately.

---

## Problems and fixes

Every error in this project says which step failed and what to do. Common ones:

| You see | Cause | Fix |
| :--- | :--- | :--- |
| Sidebar: **Backend unreachable** | The server is not running | Start it (step 6) |
| `[MODEL] expected a saved Markov model (run: python -m backend.ml.pipeline), got no file` | Models not trained yet | Run step 5 |
| `[MODEL] expected scikit-learn 1.9.1 ..., got <other version>` | Wrong scikit-learn installed | Re-run step 3, or re-run step 5 to retrain with the installed version |
| `[MODEL] expected scales for the checks ...` or `[MODEL] expected fields ...` | Model files saved by an older version of the code | Run step 5 |
| The server stops at startup with `[STORAGE] expected analyses stored by the current model version ...` | The database holds analyses stored by an older version of the models (for example after switching git branches) | Delete `backend\toc_workflow.db` and start again (it is recreated empty) |
| The dashboard looks like another version, or shows a validation error, after switching git branches | `frontend/dist` was built on the other branch (it is not in git) | Run step 4 again (`npm run build`), then reload with **Ctrl+Shift+R** |
| `[LOAD] expected an existing CSV file ...` (Overview stream / Analyze examples) | Dataset not generated | Run step 5 |
| `[STORAGE] expected every column of 'workflow_analyses' (an older database file? ...)` | A database file from an older version | Stop the server, delete `backend\toc_workflow.db`, start again (it is recreated empty) |
| Opening http://127.0.0.1:8000/ gives a JSON `NOT_FOUND` error | The dashboard was not built | Run step 4 |
| `[Errno 10048]` / "address already in use" | Port 8000 is taken (e.g. the app is already running) | Close the other window, or use `--port 8001` and open http://127.0.0.1:8001/ |
| The page looks old after an update | The browser kept the old version | Press **Ctrl+Shift+R** |

## What is where

| Folder / file | Contents |
| :--- | :--- |
| `backend/` | Python: automaton, ML, API, database, tests |
| `frontend/` | React dashboard |
| `run_tests.py` | Runs every test |
| `README.md` | Project overview and results |
| `docs/VIVA_GUIDE.md` | How the codebase produces its results (viva preparation) |
| `backend/docs/` | Workflow (FSM) specification, dataset specification, API guide |
