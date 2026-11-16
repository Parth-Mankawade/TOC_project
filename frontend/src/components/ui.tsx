import { Component, useEffect, type ErrorInfo, type ReactNode } from "react";
import { createPortal } from "react-dom";
import { ApiError } from "../api";
import type { Measurement, Verdict } from "../types";
import { VERDICTS } from "../lib/workflow";
import { formatMeasurementValue } from "../lib/format";
import { Icon, type IconName } from "./Icon";

export function PageHeader({ eyebrow, title, description, actions }: {
  eyebrow: string;
  title: ReactNode;
  description?: ReactNode;
  actions?: ReactNode;
}) {
  return (
    <header className="page-head">
      <div>
        <div className="eyebrow">{eyebrow}</div>
        <h1 className="page-title">{title}</h1>
        {description && <p className="page-desc">{description}</p>}
      </div>
      {actions && <div className="actions">{actions}</div>}
    </header>
  );
}

const TONE_ICON: Record<string, IconName> = { ok: "check", bad: "x", warn: "alert", info: "clock" };

export function VerdictBadge({ verdict, large }: { verdict: Verdict; large?: boolean }) {
  const meta = VERDICTS[verdict];
  return (
    <span className={`badge tone-${meta.tone}${large ? " lg" : ""}`}>
      <Icon name={TONE_ICON[meta.tone]} size={large ? 16 : 13} />
      {meta.label}
    </span>
  );
}

export function SeverityBadge({ severity }: { severity: string }) {
  const tone = severity === "CRITICAL" ? "bad" : severity === "WARNING" ? "warn" : "info";
  return <span className={`badge tone-${tone}`}>{severity}</span>;
}

export function PathChips({ events, max = 6, failedAt }: { events: string[]; max?: number; failedAt?: number }) {
  const shown = events.slice(0, max);
  return (
    <span className="path">
      {shown.map((e, i) => (
        <span key={i} className="path">
          {i > 0 && <span className="arrow">→</span>}
          <span className={`chip${failedAt === i + 1 ? " bad" : ""}`}>{e}</span>
        </span>
      ))}
      {events.length > max && <span className="faint">+{events.length - max}</span>}
    </span>
  );
}

export function Loading({ height = 120 }: { height?: number }) {
  return <div className="skeleton" style={{ height }} aria-label="Loading" />;
}

/** Shows the whole error: the message, and every detail line the backend sent. */
export function ErrorBox({ error, onRetry }: { error: Error; onRetry?: () => void }) {
  const details = error instanceof ApiError && error.details.length > 1 ? error.details : [];
  return (
    <div className="error-box" role="alert">
      <div style={{ display: "flex", justifyContent: "space-between", gap: 12, alignItems: "flex-start" }}>
        <span className="mono" style={{ fontSize: 12.5, wordBreak: "break-word" }}>{error.message}</span>
        {onRetry && (
          <button className="btn sm" onClick={onRetry}>
            <Icon name="refresh" size={14} /> Retry
          </button>
        )}
      </div>
      {details.length > 0 && (
        <ul className="mono" style={{ margin: "8px 0 0", paddingLeft: 18, fontSize: 12 }}>
          {details.map((d, i) => <li key={i}>{d}</li>)}
        </ul>
      )}
    </div>
  );
}

/** A rendering error shows its message instead of a blank page. */
export class ErrorBoundary extends Component<{ children: ReactNode }, { error: Error | null }> {
  state: { error: Error | null } = { error: null };

  static getDerivedStateFromError(error: Error) {
    return { error };
  }

  componentDidCatch(error: Error, info: ErrorInfo) {
    console.error(error, info.componentStack);
  }

  render() {
    if (this.state.error) {
      return (
        <div className="page">
          <ErrorBox error={this.state.error} onRetry={() => this.setState({ error: null })} />
        </div>
      );
    }
    return this.props.children;
  }
}

export function Bar({ value, tone }: { value: number; tone?: string }) {
  return (
    <div className={tone === undefined ? "bar" : `bar ${tone}`}>
      <span style={{ width: `${Math.max(0, Math.min(1, value)) * 100}%` }} />
    </div>
  );
}

/** Position of a value relative to its normal range, on a log scale so seconds and days fit together. */
export function RangeBar({ m }: { m: Measurement }) {
  const f = (v: number) => Math.log10(Math.max(v, 0) + 1);
  const lo = f(m.normal_low);
  const hi = f(m.normal_high);
  const v = f(m.value);
  const pad = 0.15 * (hi > lo ? hi - lo : 1);
  const min = Math.min(lo, v) - pad;
  const max = Math.max(hi, v) + pad;
  const pct = (x: number) => ((x - min) / (max - min)) * 100;
  return (
    <div className="range" title={`normal ${formatMeasurementValue(m.unit, m.normal_low)} – ${formatMeasurementValue(m.unit, m.normal_high)}`}>
      <div className="track" />
      <div className="band" style={{ left: `${pct(lo)}%`, width: `${Math.max(pct(hi) - pct(lo), 1.5)}%` }} />
      <div className={`marker${m.is_unusual ? " unusual" : ""}`} style={{ left: `${pct(v)}%` }} />
    </div>
  );
}

export function Drawer({ title, onClose, children }: { title: ReactNode; onClose: () => void; children: ReactNode }) {
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => e.key === "Escape" && onClose();
    window.addEventListener("keydown", onKey);
    document.body.style.overflow = "hidden";
    return () => {
      window.removeEventListener("keydown", onKey);
      document.body.style.overflow = "";
    };
  }, [onClose]);
  // Rendered on <body>: inside an animated (transformed) page, position: fixed would be relative to the page.
  return createPortal(
    <>
      <div className="overlay" onClick={onClose} />
      <aside className="drawer" role="dialog" aria-modal="true">
        <div className="drawer-head">
          <strong>{title}</strong>
          <button className="btn ghost sm" onClick={onClose} aria-label="Close">
            <Icon name="x" size={16} /> Close
          </button>
        </div>
        <div className="drawer-body">{children}</div>
      </aside>
    </>,
    document.body,
  );
}

export function Donut({ parts, size = 168 }: { parts: { value: number; color: string }[]; size?: number }) {
  const total = parts.reduce((s, p) => s + p.value, 0);
  const r = 62;
  const c = 2 * Math.PI * r;
  let offset = 0;
  return (
    <svg width={size} height={size} viewBox="0 0 160 160" role="img" aria-label="Verdict distribution">
      <circle cx="80" cy="80" r={r} fill="none" stroke="var(--surface-3)" strokeWidth="18" />
      {total > 0 && parts.map((p, i) => {
        const len = (p.value / total) * c;
        const el = (
          <circle key={i} cx="80" cy="80" r={r} fill="none" stroke={p.color} strokeWidth="18"
            strokeDasharray={`${Math.max(len - 2, 0)} ${c}`} strokeDashoffset={-offset} transform="rotate(-90 80 80)" />
        );
        offset += len;
        return el;
      })}
    </svg>
  );
}
