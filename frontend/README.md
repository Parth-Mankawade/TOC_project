# Workflow Monitor: dashboard

React + TypeScript (Vite). Every number it shows comes from the backend API; the frontend has no
hand-written results or samples.

## Pages

| Page | What it shows |
| :--- | :--- |
| **Overview** | The **live demo stream** (100 real test orders sent one at a time, each path drawn on the automaton, counters comparing verdicts with how the generator made each order), totals per verdict, recent analyses. |
| **Analyze** | Real test orders of every kind (normal, five anomaly types, six violation types, unfinished), or build an order event by event. The result shows *How the verdict was reached* (each check lighting up in order) and a **playable execution path** (play, pause, step, speed). |
| **History** | Every stored analysis, filters by verdict and status, lookup by order ID. |
| **Workflow & reachability** | The automaton M = (Q, Σ, δ, q₀, F), reachability facts, and *Play this path* for the shortest route to any state. |
| **Models & results** | The held-out results, the validation choice of timing model, the data check, the Markov probabilities on the diagram, the timing model's normal ranges. |

Errors are shown where the data would be, with the backend's own message
(`[CHECKPOINT] expected ..., got ...; input=...`) and every detail line; the sidebar shows whether the
backend is ready and, if not, why.

## Commands (from this folder)

```powershell
npm install        # once
npm run build      # creates dist/, served by the backend at /app/
npm test           # Vitest unit tests
npm run typecheck  # TypeScript only
npm run dev        # http://localhost:5173/app/ with the backend on port 8000
```

## Structure

```text
src/
  App.tsx, main.tsx     navigation, backend status, error boundary
  api.ts, types.ts      typed API client (fails loudly) and response shapes
  lib/                  workflow replay, playback frames and analysis stages, formatting, hooks
  components/           FsmDiagram, PathPlayer, AnalysisFlow, ResultView, ui
  pages/                Overview, Analyze, History, Workflow, Models
  test/                 fixtures: real backend output (fsm.json, records.json)
```
