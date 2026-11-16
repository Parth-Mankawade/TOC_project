import { useEffect, useState } from "react";
import { HashRouter, NavLink, Navigate, Route, Routes, useNavigate } from "react-router-dom";
import { ApiError, api } from "./api";
import { Icon, type IconName } from "./components/Icon";
import { ErrorBoundary } from "./components/ui";
import { FsmContext, useAsync } from "./lib/hooks";
import { DETECTOR_LABELS } from "./lib/workflow";
import { Analyze } from "./pages/Analyze";
import { History } from "./pages/History";
import { Models } from "./pages/Models";
import { Overview } from "./pages/Overview";
import { Workflow } from "./pages/Workflow";
import type { DetectorName, Readiness } from "./types";

const NAV: { to: string; label: string; icon: IconName; key: string }[] = [
  { to: "/", label: "Overview", icon: "overview", key: "1" },
  { to: "/analyze", label: "Analyze", icon: "analyze", key: "2" },
  { to: "/history", label: "History", icon: "history", key: "3" },
  { to: "/workflow", label: "Workflow & reachability", icon: "workflow", key: "4" },
  { to: "/models", label: "Models & results", icon: "models", key: "5" },
];
const READINESS_POLL_MS = 20000;

function Shortcuts() {
  // Keys 1-5 jump between pages (ignored while typing in a field).
  const navigate = useNavigate();
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      const target = e.target as HTMLElement;
      if (e.metaKey || e.ctrlKey || e.altKey || /INPUT|TEXTAREA|SELECT/.test(target.tagName)) return;
      const item = NAV.find((n) => n.key === e.key);
      if (item) navigate(item.to);
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [navigate]);
  return null;
}

type Status = { kind: "checking" } | { kind: "ready"; info: Readiness } | { kind: "error"; error: ApiError };

function BackendStatus() {
  const [status, setStatus] = useState<Status>({ kind: "checking" });
  useEffect(() => {
    const check = () => api.ready()
      .then((info) => setStatus({ kind: "ready", info }))
      .catch((e: ApiError) => setStatus({ kind: "error", error: e }));
    check();
    const id = window.setInterval(check, READINESS_POLL_MS);
    return () => window.clearInterval(id);
  }, []);

  if (status.kind === "checking") return <div className="status-card"><strong><span className="dot" /> Checking backend…</strong></div>;
  if (status.kind === "ready") {
    const timing = DETECTOR_LABELS[status.info.timing_model.model_name as DetectorName];
    return (
      <div className="status-card">
        <strong><span className="dot ok" /> Backend ready</strong>
        <span className="faint">Database ok · Markov chain + {timing} loaded</span>
      </div>
    );
  }
  return (
    <div className="status-card">
      <strong><span className="dot bad" /> {status.error.status === 0 ? "Backend unreachable" : `Backend not ready (${status.error.status})`}</strong>
      <span className="faint mono" style={{ fontSize: 11, wordBreak: "break-word" }}>{status.error.message}</span>
    </div>
  );
}

export default function App() {
  const fsm = useAsync(() => api.fsm());

  return (
    <HashRouter>
      <Shortcuts />
      <FsmContext.Provider value={fsm}>
        <div className="shell">
          <aside className="sidebar">
            <div className="brand">
              <span className="orb" />
              <div>
                <div className="brand-name">Workflow Monitor</div>
                <div className="brand-sub">Automata + ML · PS9</div>
              </div>
            </div>
            <p className="sidebar-intro">
              Checks online-shop orders with a finite state machine and two machine learning models, and explains every verdict.
            </p>
            <div className="nav-label">Browse</div>
            <nav className="nav" aria-label="Main">
              {NAV.map((n) => (
                <NavLink key={n.to} to={n.to} end={n.to === "/"}>
                  <Icon name={n.icon} />
                  <span>{n.label}</span>
                  <span className="kbd">{n.key}</span>
                </NavLink>
              ))}
            </nav>
            <div className="nav-label">Resources</div>
            <nav className="nav" aria-label="Resources">
              <a href="/docs" target="_blank" rel="noreferrer">
                <Icon name="docs" />
                <span>API docs</span>
                <span className="kbd"><Icon name="external" size={13} /></span>
              </a>
            </nav>
            <div className="sidebar-foot"><BackendStatus /></div>
          </aside>

          <main className="main">
            <ErrorBoundary>
              <Routes>
                <Route path="/" element={<Overview />} />
                <Route path="/analyze" element={<Analyze />} />
                <Route path="/history" element={<History />} />
                <Route path="/workflow" element={<Workflow />} />
                <Route path="/models" element={<Models />} />
                <Route path="*" element={<Navigate to="/" replace />} />
              </Routes>
            </ErrorBoundary>
          </main>
        </div>
      </FsmContext.Provider>
    </HashRouter>
  );
}
