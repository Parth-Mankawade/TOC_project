import { useEffect, useMemo, useState } from "react";
import type { AnalysisRecord } from "../types";
import { analysisStages, type StageStatus } from "../lib/playback";
import { Icon, type IconName } from "./Icon";

const STAGE_MS = 500; // time between two stages lighting up

const STATUS: Record<StageStatus, { label: string; icon: IconName }> = {
  pass: { label: "passed", icon: "check" },
  fail: { label: "failed", icon: "x" },
  waiting: { label: "in progress", icon: "clock" },
  clear: { label: "not flagged", icon: "check" },
  flagged: { label: "flagged", icon: "alert" },
  skipped: { label: "skipped", icon: "info" },
};

/** The checks in the order the backend ran them, revealed one by one. */
export function AnalysisFlow({ record }: { record: AnalysisRecord }) {
  const stages = useMemo(() => analysisStages(record), [record]);
  const [shown, setShown] = useState(0);
  const [run, setRun] = useState(0);

  useEffect(() => {
    setShown(0);
    const id = window.setInterval(() => setShown((s) => (s < stages.length ? s + 1 : s)), STAGE_MS);
    return () => window.clearInterval(id);
  }, [stages, run]);

  return (
    <section className="card">
      <div className="card-head">
        <div>
          <h3 className="card-title">How the verdict was reached</h3>
          <p className="card-sub">Each check in the order the backend runs it. A broken rule or an unfinished order stops the ML checks.</p>
        </div>
        <button className="btn sm" onClick={() => setRun(run + 1)} disabled={shown < stages.length}>
          <Icon name="restart" size={14} /> Replay
        </button>
      </div>
      <ol className="flow">
        {stages.map((s, i) => (
          <li key={s.key} className={`stage st-${s.status}${i < shown ? " revealed" : ""}`}>
            <span className="stage-icon"><Icon name={STATUS[s.status].icon} size={16} /></span>
            <div className="stage-title">{s.title}</div>
            <div className="stage-status">{STATUS[s.status].label}</div>
            <div className="stage-detail">{s.detail}</div>
          </li>
        ))}
      </ol>
    </section>
  );
}
