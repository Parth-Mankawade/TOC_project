import type { AnalysisRecord, FsmDefinition, Measurement } from "../types";
import { describePercentile, describeProbability, formatAmount, formatDateTime, formatDuration, formatMeasurementValue } from "../lib/format";
import { useFsmState } from "../lib/hooks";
import { DETECTOR_LABELS, eventsOf, possibleOutcomes, walkPath } from "../lib/workflow";
import { AnalysisFlow } from "./AnalysisFlow";
import { Icon } from "./Icon";
import { PathPlayer, type StepFacts } from "./PathPlayer";
import { Bar, ErrorBox, Loading, RangeBar, SeverityBadge, VerdictBadge } from "./ui";

/** Per-step model output, keyed by step number. */
function stepFacts(record: AnalysisRecord): Map<number, StepFacts> {
  const facts = new Map<number, StepFacts>();
  if (record.ml_evaluation === null) return facts;
  const durations = new Map<number, Measurement>();
  if (record.timing_evaluation !== null) {
    for (const m of record.timing_evaluation.measurements) if (m.step !== null) durations.set(m.step, m);
  }
  for (const t of record.ml_evaluation.transitions) {
    const d = durations.get(t.step);
    facts.set(t.step, {
      probability: t.probability,
      rareStep: t.is_unusual,
      unusual: t.is_unusual || (d !== undefined && d.is_unusual),
      duration: d === undefined ? null : d,
    });
  }
  return facts;
}

export function ResultView({ record, autoPlay = true }: { record: AnalysisRecord; autoPlay?: boolean }) {
  const fsm = useFsmState();
  const events = eventsOf(record);
  const facts = stepFacts(record);
  const ml = record.ml_evaluation;
  const timing = record.timing_evaluation;
  const err = record.fsm_evaluation.error;

  return (
    <div className="grid" style={{ gap: 16 }}>
      <section className="card">
        <div className="verdict-hero">
          <div>
            <div className="actions" style={{ alignItems: "center" }}>
              <VerdictBadge verdict={record.verdict} large />
              <SeverityBadge severity={record.diagnostics.severity} />
              <span className="faint" style={{ fontSize: 12.5 }}>
                {record.order_id !== null ? <span className="mono">{record.order_id}</span> : "No order ID"} · analysis #{record.id} ·{" "}
                {formatDateTime(record.created_at)}
              </span>
            </div>
            <h2>{record.summary}</h2>
          </div>
          <div className="actions" style={{ gap: 10 }}>
            {ml !== null && <ScoreTile label="Path score" value={ml.anomaly_score} flagged={ml.is_ml_anomaly} />}
            {timing !== null && <ScoreTile label="Timing score" value={timing.anomaly_score} flagged={timing.is_timing_anomaly} />}
            {err !== null && (
              <div className="stat" style={{ minWidth: 150 }}>
                <span className="stat-label">Failed at step</span>
                <span className="stat-value" style={{ color: "var(--bad)" }}>{err.failed_step}</span>
              </div>
            )}
          </div>
        </div>
      </section>

      <section className="card">
        <div className="card-head">
          <div>
            <h3 className="card-title">Execution path</h3>
            <p className="card-sub">
              Press play to walk the order through the automaton. <span style={{ color: "var(--warn)" }}>Amber</span> marks an
              unusual step, <span style={{ color: "var(--bad)" }}>red</span> the step that broke a rule.
            </p>
          </div>
          {record.order_amount !== null && <span className="badge tone-info">Amount {formatAmount(record.order_amount)}</span>}
        </div>
        {fsm.error !== null ? <ErrorBox error={fsm.error} onRetry={fsm.reload} />
          : fsm.data === null ? <Loading height={260} />
            : <PathPlayer fsm={fsm.data} events={events} facts={facts} autoPlay={autoPlay} />}
        {fsm.data !== null && <StepTable record={record} fsm={fsm.data} events={events} facts={facts} />}
      </section>

      <AnalysisFlow record={record} />

      {err !== null ? (
        <section className="card" style={{ borderColor: "var(--bad-line)" }}>
          <div className="check-head">
            <span className="check-icon tone-bad"><Icon name="x" /></span>
            <div>
              <h3 className="card-title">Rule check failed: {err.error_type.replace(/_/g, " ").toLowerCase()}</h3>
              <p className="card-sub">The finite state machine stopped here, so neither ML check ran.</p>
            </div>
          </div>
          <dl className="kv">
            <dt>Step</dt><dd>{err.failed_step}</dd>
            <dt>Event</dt><dd><span className="chip bad">{err.failed_event}</span></dd>
            <dt>State</dt><dd className="mono">{err.state_at_failure}</dd>
            <dt>Allowed here</dt>
            <dd className="path">
              {err.allowed_events.length > 0 ? err.allowed_events.map((e) => <span key={e} className="chip">{e}</span>)
                : <span className="faint">nothing: the order had already finished</span>}
            </dd>
            <dt>FSM message</dt><dd className="mono" style={{ fontSize: 12 }}>{err.message}</dd>
          </dl>
        </section>
      ) : (
        <div className="grid g-2">
          <PathCheck record={record} />
          <TimingCheck record={record} />
        </div>
      )}

      {record.verdict === "VALID_INCOMPLETE" && fsm.data !== null && (
        <section className="card">
          <h3 className="card-title">What can still happen</h3>
          <p className="card-sub" style={{ marginBottom: 14 }}>
            Reachability from <span className="mono">{record.fsm_evaluation.final_state}</span>: every final state the order
            can still reach, with the shortest way there.
          </p>
          <ul className="list-plain">
            {Object.entries(possibleOutcomes(record.fsm_evaluation.final_state, fsm.data)).map(([state, path]) => (
              <li key={state} className="factor" style={{ alignItems: "center" }}>
                <span className="mono" style={{ minWidth: 110, fontWeight: 600 }}>{state}</span>
                <span className="path">
                  {path.map((e, i) => (
                    <span key={i} className="path">{i > 0 && <span className="arrow">→</span>}<span className="chip">{e}</span></span>
                  ))}
                </span>
              </li>
            ))}
          </ul>
        </section>
      )}

      <section className="card">
        <h3 className="card-title">Why</h3>
        <p style={{ margin: "8px 0 14px", fontSize: 15 }}>{record.diagnostics.root_cause}</p>
        <ul className="list-plain">
          {record.diagnostics.key_factors.map((f, i) => (
            <li key={i} className={`factor${/unusual|combination/i.test(f) ? " unusual" : ""}`}>
              <span className="mark" />
              <span>{f}</span>
            </li>
          ))}
        </ul>
        <div className="callout accent" style={{ marginTop: 14 }}>
          <Icon name="arrow" />
          <div>
            <strong>Recommended action</strong>
            <div>{record.diagnostics.recommended_action}</div>
          </div>
        </div>
      </section>
    </div>
  );
}

function StepTable({ record, fsm, events, facts }: {
  record: AnalysisRecord; fsm: FsmDefinition; events: string[]; facts: Map<number, StepFacts>;
}) {
  const walk = walkPath(events, fsm);
  const err = record.fsm_evaluation.error;
  return (
    <div className="steps" style={{ marginTop: 12 }}>
      {events.map((event, i) => {
        const step = i + 1;
        const w = walk.find((s) => s.step === step);
        const f = facts.get(step);
        const failed = err !== null && err.failed_step === step;
        const tone = failed ? "bad" : f !== undefined && f.unusual ? "warn" : "";
        return (
          <div className="step-row" key={step} style={w === undefined ? { opacity: 0.45 } : undefined}>
            <span className={`step-num ${tone}`}>{step}</span>
            <div>
              <span className={`chip ${tone}`}>{event}</span>
              <div className="step-states" style={{ marginTop: 4 }}>
                {w === undefined ? "not evaluated (after the failing step)" : failed ? `not allowed from ${w.from}` : `${w.from} → ${w.to}`}
              </div>
            </div>
            <div>
              {f !== undefined ? (
                <div className="prob" title="P(event | state) learned from normal orders">
                  <Bar value={f.probability} tone={f.unusual ? "warn" : undefined} />
                  <span>{f.probability.toFixed(3)}</span>
                </div>
              ) : (
                <span className="faint" style={{ fontSize: 12 }}>
                  {failed ? "rule broken here" : w === undefined ? "" : "path check did not run"}
                </span>
              )}
            </div>
            <div style={{ fontSize: 12.5 }}>
              {f !== undefined && f.duration !== null ? (
                <>
                  <div style={{ display: "flex", justifyContent: "space-between", gap: 8 }}>
                    <span style={f.duration.is_unusual ? { color: "var(--warn)", fontWeight: 600 } : undefined}>
                      took {formatDuration(f.duration.value)}
                    </span>
                    <span className="faint">{describePercentile(f.duration.percentile, "seconds")}</span>
                  </div>
                  <RangeBar m={f.duration} />
                </>
              ) : (
                <span className="faint">{record.event_timestamps !== null && i === 0 ? `at ${formatDateTime(record.event_timestamps[0])}` : ""}</span>
              )}
            </div>
          </div>
        );
      })}
    </div>
  );
}

function ScoreTile({ label, value, flagged }: { label: string; value: number; flagged: boolean }) {
  return (
    <div className="stat" style={{ minWidth: 150, borderColor: flagged ? "var(--warn-line)" : undefined }}>
      <span className="stat-label"><span className={`swatch ${flagged ? "sw-warn" : "sw-ok"}`} /> {label}</span>
      <span className="stat-value">{value.toFixed(2)}</span>
      <span className="stat-note">{flagged ? "above 0: flagged" : "0 or below: not flagged"}</span>
    </div>
  );
}

function PathCheck({ record }: { record: AnalysisRecord }) {
  const ml = record.ml_evaluation;
  return (
    <section className="card">
      <div className="check-head">
        <span className={`check-icon ${ml === null ? "tone-info" : ml.is_ml_anomaly ? "tone-warn" : "tone-ok"}`}><Icon name="route" /></span>
        <div>
          <h3 className="card-title">Event order · Markov chain</h3>
          <p className="card-sub">How likely is this exact sequence of events in normal orders?</p>
        </div>
      </div>
      {ml !== null ? (
        <>
          <div className="score"><b>{describeProbability(ml.path_log10_probability)}</b></div>
          <p className="muted" style={{ margin: "8px 0 0" }}>
            {ml.is_ml_anomaly ? "Flagged: " : "Not flagged: "}paths rarer than{" "}
            {describeProbability(ml.path_log10_probability + ml.anomaly_score)} are flagged.
          </p>
          <dl className="kv" style={{ marginTop: 14 }}>
            <dt>Path score</dt><dd>{ml.anomaly_score.toFixed(2)} <span className="faint">(each +1 = 10× rarer than the threshold)</span></dd>
            <dt>Unusual steps</dt><dd>{ml.transitions.filter((t) => t.is_unusual).length}</dd>
          </dl>
        </>
      ) : (
        <p className="muted">{record.ml_skip_reason}</p>
      )}
    </section>
  );
}

function TimingCheck({ record }: { record: AnalysisRecord }) {
  const timing = record.timing_evaluation;
  return (
    <section className="card">
      <div className="check-head">
        <span className={`check-icon ${timing === null ? "tone-info" : timing.is_timing_anomaly ? "tone-warn" : "tone-ok"}`}><Icon name="clock" /></span>
        <div>
          <h3 className="card-title">Timing & amount{timing !== null && ` · ${DETECTOR_LABELS[timing.model_name]}`}</h3>
          <p className="card-sub">Are the times, the amount and the outcome unusual together?</p>
        </div>
      </div>
      {timing !== null ? (
        <>
          <p className="muted" style={{ margin: "0 0 12px" }}>
            {timing.is_timing_anomaly ? "Flagged" : "Within normal behaviour"} · score {timing.anomaly_score.toFixed(2)}
          </p>
          <ul className="list-plain">
            {timing.measurements.filter((m) => m.step === null).map((m) => (
              <li key={m.name}>
                <div style={{ display: "flex", justifyContent: "space-between", fontSize: 12.5, gap: 8 }}>
                  <span style={m.is_unusual ? { color: "var(--warn)", fontWeight: 600 } : undefined}>
                    {m.name.replace(/_/g, " ")}: {formatMeasurementValue(m.unit, m.value)}
                  </span>
                  <span className="faint">{describePercentile(m.percentile, m.unit)}</span>
                </div>
                <RangeBar m={m} />
              </li>
            ))}
          </ul>
          <p className="faint" style={{ margin: "10px 0 0", fontSize: 12 }}>Step durations are shown next to each step above.</p>
        </>
      ) : (
        <div className="callout">
          <Icon name="info" />
          <div>{record.timing_skip_reason}</div>
        </div>
      )}
    </section>
  );
}
