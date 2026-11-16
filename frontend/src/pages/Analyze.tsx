import { useMemo, useRef, useState } from "react";
import { api, exampleRequest } from "../api";
import { Icon } from "../components/Icon";
import { ResultView } from "../components/ResultView";
import { Drawer, ErrorBox, Loading, PageHeader, PathChips, VerdictBadge } from "../components/ui";
import { formatAmount, formatDuration, formatPercent } from "../lib/format";
import { useAsync, useFsmState } from "../lib/hooks";
import {
  ANOMALY_TYPE_LABELS, EVENTS, EVENT_LABELS, VERDICTS, VERDICT_ORDER, VIOLATION_TYPE_LABELS, eventsOf, labelOf, walkPath,
} from "../lib/workflow";
import type {
  AnalysisRecord, AnalyzeRequest, BatchResponse, EvaluatedModel, EvaluationReport, ExampleOrder, OrderCategory, TimingModelInfo, Verdict,
} from "../types";

const UNITS = { s: 1, min: 60, h: 3600, days: 86400 } as const;
type Unit = keyof typeof UNITS;
interface Gap { value: number; unit: Unit }

// A pair of events the workflow does not allow has no learned typical time; new steps of that
// kind start at this many seconds (the order will be a rule violation, so timing is not scored).
const UNLEARNED_GAP_SECONDS = 60;
const START_DAYS_AGO = 2; // the form's first event starts this long ago

const EXPECTED: Record<OrderCategory, Verdict> = {
  NORMAL: "NORMAL", STATISTICAL_ANOMALY: "STATISTICAL_ANOMALY", RULE_VIOLATION: "RULE_VIOLATION", IN_PROGRESS: "VALID_INCOMPLETE",
};

function toGap(seconds: number): Gap {
  if (seconds >= 2 * 86400) return { value: +(seconds / 86400).toFixed(2), unit: "days" };
  if (seconds >= 3600) return { value: +(seconds / 3600).toFixed(2), unit: "h" };
  if (seconds >= 60) return { value: +(seconds / 60).toFixed(1), unit: "min" };
  return { value: Math.max(1, Math.round(seconds)), unit: "s" };
}

function typicalGap(timing: TimingModelInfo, previous: string, event: string): Gap {
  const name = `${previous} -> ${event}`;
  return toGap(name in timing.steps ? timing.steps[name].median : UNLEARNED_GAP_SECONDS);
}

function localInputValue(ms: number): string {
  const d = new Date(ms);
  d.setSeconds(0, 0);
  return new Date(d.getTime() - d.getTimezoneOffset() * 60000).toISOString().slice(0, 16);
}

function generatedAs(o: ExampleOrder): string {
  if (o.category === "STATISTICAL_ANOMALY") return labelOf(ANOMALY_TYPE_LABELS, o.anomaly_type);
  if (o.category === "RULE_VIOLATION") return labelOf(VIOLATION_TYPE_LABELS, o.anomaly_type);
  if (o.category === "IN_PROGRESS") return "Unfinished order";
  return `Normal order (${o.event_sequence[o.event_sequence.length - 1].replace("_", " ").toLowerCase()})`;
}

const GROUPS: { category: OrderCategory; title: string; tone: string }[] = [
  { category: "NORMAL", title: "Normal", tone: "ok" },
  { category: "STATISTICAL_ANOMALY", title: "Anomalies (legal but suspicious)", tone: "warn" },
  { category: "RULE_VIOLATION", title: "Rule violations", tone: "bad" },
  { category: "IN_PROGRESS", title: "In progress", tone: "info" },
];

export function Analyze() {
  const [tab, setTab] = useState<"single" | "batch">("single");
  const examples = useAsync(() => api.examples());
  const models = useAsync(() => api.models());
  return (
    <div className="page">
      <PageHeader
        eyebrow="Analyze"
        title={<>Check an <em>order</em></>}
        description="Pick a real order from the held-out test data, or build one event by event. Add timestamps and the amount to also run the timing check."
        actions={
          <div className="tabs" role="tablist">
            <button role="tab" aria-selected={tab === "single"} className={tab === "single" ? "active" : ""} onClick={() => setTab("single")}>Single order</button>
            <button role="tab" aria-selected={tab === "batch"} className={tab === "batch" ? "active" : ""} onClick={() => setTab("batch")}>Batch</button>
          </div>
        }
      />
      {examples.error !== null && <div style={{ marginBottom: 16 }}><ErrorBox error={examples.error} onRetry={examples.reload} /></div>}
      {models.error !== null && <div style={{ marginBottom: 16 }}><ErrorBox error={models.error} onRetry={models.reload} /></div>}
      {examples.data === null || models.data === null ? (
        examples.error === null && models.error === null ? <Loading height={360} /> : null
      ) : tab === "single" ? (
        <SingleOrder examples={examples.data} timing={models.data.timing_model} hybrid={hybridResult(models.data.evaluation)} />
      ) : (
        <Batch examples={examples.data} />
      )}
    </div>
  );
}

function hybridResult(report: EvaluationReport): EvaluatedModel {
  const hybrid = report.test.models.find((m) => m.model === "hybrid");
  if (hybrid === undefined) throw new Error(`[UI] expected a 'hybrid' model in the evaluation report, got ${report.test.models.map((m) => m.model).join(", ")}; input=GET /api/v1/models`);
  return hybrid;
}

function SingleOrder({ examples, timing, hybrid }: { examples: ExampleOrder[]; timing: TimingModelInfo; hybrid: EvaluatedModel }) {
  const fsmState = useFsmState();
  const fsm = fsmState.data;
  const [example, setExample] = useState<ExampleOrder | null>(null);
  const [orderId, setOrderId] = useState("");
  const [order, setOrder] = useState<{ events: string[]; gaps: Gap[] }>({ events: [], gaps: [] });
  const [withTiming, setWithTiming] = useState(true);
  const [start, setStart] = useState(() => localInputValue(Date.now() - START_DAYS_AGO * 86400 * 1000));
  const [amount, setAmount] = useState(String(timing.amount.median.toFixed(2)));
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<Error | null>(null);
  const [result, setResult] = useState<AnalysisRecord | null>(null);
  const resultRef = useRef<HTMLDivElement>(null);

  const { events, gaps } = order;
  /** Replace the events; kept gaps stay, new steps get the learned typical time. */
  const setEvents = (next: string[]) => {
    setExample(null);
    setOrder({ events: next, gaps: next.slice(1).map((e, i) => (i < gaps.length && next[i] === events[i] && next[i + 1] === events[i + 1]
      ? gaps[i] : typicalGap(timing, next[i], e))) });
  };

  const walk = useMemo(() => (fsm === null ? [] : walkPath(events, fsm)), [events, fsm]);
  const last = walk.length > 0 ? walk[walk.length - 1] : null;
  const broken = last !== null && !last.ok;
  const state = fsm === null ? null : last === null ? fsm.initial_state : last.ok && last.to !== null ? last.to : last.from;
  const allowedNext = new Set(fsm !== null && state !== null && !broken && !fsm.terminal_states.includes(state)
    ? fsm.transitions.filter((t) => t.from_state === state).map((t) => t.event) : []);

  const loadExample = (o: ExampleOrder) => {
    const times = o.event_timestamps.map((t) => new Date(t).getTime());
    setExample(o);
    setOrderId(o.order_id);
    setOrder({ events: o.event_sequence, gaps: times.slice(1).map((t, i) => toGap((t - times[i]) / 1000)) });
    setStart(localInputValue(times[0]));
    setAmount(String(o.order_amount));
    setWithTiming(true);
    setResult(null);
    setError(null);
  };

  const submit = async () => {
    setBusy(true);
    setError(null);
    let body: AnalyzeRequest;
    if (example !== null && withTiming) {
      body = exampleRequest(example); // exactly the stored timestamps and amount
    } else {
      body = { event_sequence: events };
      if (orderId.trim()) body.order_id = orderId.trim();
      if (withTiming) {
        let t = new Date(start).getTime();
        const stamps = [new Date(t).toISOString()];
        for (const g of gaps) {
          t += g.value * UNITS[g.unit] * 1000;
          stamps.push(new Date(t).toISOString());
        }
        body.event_timestamps = stamps;
        body.order_amount = Number(amount);
      }
    }
    try {
      setResult(await api.analyze(body));
      window.setTimeout(() => resultRef.current?.scrollIntoView({ behavior: "smooth", block: "start" }), 50);
    } catch (e) {
      setError(e as Error);
    } finally {
      setBusy(false);
    }
  };

  return (
    <>
      <section className="card tint" style={{ marginBottom: 16 }}>
        <div className="card-head">
          <div>
            <h3 className="card-title">Start from a real test order</h3>
            <p className="card-sub">
              Orders from the held-out test split, with their real timestamps and amount. The label says how the data generator
              made each one, so you can compare it with the system's verdict.
            </p>
          </div>
        </div>
        <div className="example-groups">
          {GROUPS.map((g) => (
            <div key={g.category}>
              <div className="label" style={{ marginBottom: 8 }}><span className={`swatch sw-${g.tone}`} /> {g.title}</div>
              <div className="sample-grid">
                {examples.filter((o) => o.category === g.category).map((o) => (
                  <button key={o.order_id} className={`sample${example !== null && example.order_id === o.order_id ? " active" : ""}`} onClick={() => loadExample(o)}>
                    <strong>{generatedAs(o)}</strong>
                    <span className="mono">{o.order_id} · {o.event_sequence.length} events · {formatAmount(o.order_amount)}</span>
                  </button>
                ))}
              </div>
            </div>
          ))}
        </div>
      </section>

      <div className="grid g-2" style={{ alignItems: "start" }}>
        <section className="card">
          <h3 className="card-title">Events</h3>
          <p className="card-sub" style={{ marginBottom: 14 }}>
            Click events in order. A <span style={{ color: "var(--accent)" }}>blue dot</span> marks events the workflow allows next;
            the others can be clicked too, to see a rule violation.
          </p>
          <div className="token-row" aria-label="Event sequence">
            {events.length === 0 && <span className="faint" style={{ alignSelf: "center", fontSize: 13 }}>No events yet.</span>}
            {events.map((e, i) => (
              <span key={`${i}-${e}`} className={`token${broken && last !== null && last.step === i + 1 ? " bad" : ""}`}>
                <span className="n">{i + 1}</span>
                {e}
                <button aria-label={`Remove ${e}`} onClick={() => setEvents(events.filter((_, j) => j !== i))}>
                  <Icon name="x" size={13} />
                </button>
              </span>
            ))}
          </div>
          <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", margin: "8px 0 14px", fontSize: 12.5 }}>
            <span className={broken ? "bad-text" : "muted"}>
              {broken && last !== null ? `Step ${last.step} (${last.event}) is not allowed from ${last.from}.`
                : state !== null && fsm !== null ? <>Current state: <span className="mono">{state}</span>{fsm.terminal_states.includes(state) && " (finished)"}</> : ""}
            </span>
            {events.length > 0 && (
              <button className="btn ghost sm" onClick={() => { setEvents([]); setResult(null); }}>
                <Icon name="trash" size={14} /> Clear
              </button>
            )}
          </div>
          <div className="event-palette">
            {EVENTS.map((e) => (
              <button key={e} className={`event-btn${allowedNext.has(e) ? " allowed" : ""}`} title={e} onClick={() => setEvents([...events, e])}>
                {EVENT_LABELS[e]}
              </button>
            ))}
          </div>
        </section>

        <section className="card">
          <div className="card-head">
            <div>
              <h3 className="card-title">Order details</h3>
              <p className="card-sub">The order ID is optional; it groups analyses of the same order in History.</p>
            </div>
          </div>
          <div className="field">
            <label htmlFor="oid">Order ID</label>
            <input id="oid" className="input mono" value={orderId} maxLength={64} placeholder="e.g. ORD-2026-0042" onChange={(e) => setOrderId(e.target.value)} />
          </div>

          <div style={{ margin: "18px 0 12px", display: "flex", justifyContent: "space-between", alignItems: "center" }}>
            <label className="switch">
              <input type="checkbox" checked={withTiming} onChange={(e) => setWithTiming(e.target.checked)} />
              Timing & amount
            </label>
            <span className="faint" style={{ fontSize: 12 }}>{withTiming ? "runs the timing check" : "only rules + event order"}</span>
          </div>

          {withTiming && example !== null && (
            <div className="callout" style={{ marginBottom: 12 }}>
              <Icon name="info" />
              <div>Uses this order's real timestamps and amount ({formatAmount(example.order_amount)}). Change any event to edit them.</div>
            </div>
          )}
          {withTiming && example === null && (
            <div className="grid" style={{ gap: 12 }}>
              <div className="grid g-2" style={{ gap: 10 }}>
                <div className="field">
                  <label htmlFor="start">First event at</label>
                  <input id="start" type="datetime-local" className="input" value={start} onChange={(e) => setStart(e.target.value)} />
                </div>
                <div className="field">
                  <label htmlFor="amt">Order amount</label>
                  <input id="amt" type="number" min="0.01" step="0.01" className="input" value={amount} onChange={(e) => setAmount(e.target.value)} />
                </div>
              </div>
              {events.length > 1 && <div className="label">Time before each event (starts at the typical time learned from normal orders)</div>}
              {gaps.map((g, i) => (
                <div className="gap-row" key={i}>
                  <span className="step-num" style={{ width: 24, height: 24 }}>{i + 2}</span>
                  <span className="mono" style={{ fontSize: 11.5 }} title={`${events[i]} → ${events[i + 1]}`}>
                    <span className="faint">{events[i]} →</span> {events[i + 1]}
                  </span>
                  <input className="input" type="number" min="1" aria-label={`Gap before step ${i + 2}`} value={g.value}
                    onChange={(ev) => setOrder({ events, gaps: gaps.map((x, j) => (j === i ? { ...x, value: Math.max(1, Number(ev.target.value)) } : x)) })} />
                  <select className="select" aria-label={`Unit for step ${i + 2}`} value={g.unit}
                    onChange={(ev) => setOrder({ events, gaps: gaps.map((x, j) => (j === i ? { ...x, unit: ev.target.value as Unit } : x)) })}>
                    {(Object.keys(UNITS) as Unit[]).map((u) => <option key={u}>{u}</option>)}
                  </select>
                </div>
              ))}
              {gaps.length > 0 && (
                <div className="faint" style={{ fontSize: 12 }}>
                  Whole order spans {formatDuration(gaps.reduce((s, g) => s + g.value * UNITS[g.unit], 0))}.
                </div>
              )}
            </div>
          )}

          <button className="btn primary" style={{ width: "100%", marginTop: 20, height: 44 }} disabled={busy || events.length === 0} onClick={submit}>
            <Icon name={busy ? "loader" : "analyze"} className={busy ? "spin" : undefined} />
            {busy ? "Analyzing…" : events.length === 0 ? "Add at least one event" : `Analyze ${events.length} event${events.length > 1 ? "s" : ""}`}
          </button>
          {error !== null && <div style={{ marginTop: 12 }}><ErrorBox error={error} /></div>}
        </section>
      </div>

      <div ref={resultRef} style={{ scrollMarginTop: 20 }}>
        {result !== null && (
          <>
            <h2 className="section-title">Result</h2>
            {example !== null && (
              <div className={`callout ${result.verdict === EXPECTED[example.category] ? "accent" : "warn"}`} style={{ marginBottom: 16 }}>
                <Icon name={result.verdict === EXPECTED[example.category] ? "check" : "alert"} />
                <div>
                  The data generator made this order as <b>{generatedAs(example)}</b>; the system's verdict is{" "}
                  <b>{VERDICTS[result.verdict].label}</b>.{" "}
                  {result.verdict === EXPECTED[example.category] ? "They agree."
                    : example.category === "NORMAL"
                      ? `This is a false alarm. On the test split, ${formatPercent(hybrid.false_alarm_rate)} of normal orders (${hybrid.false_alarms} of ${hybrid.normal_orders}) were flagged.`
                      : `This anomaly was missed. On the test split, the system caught ${hybrid.anomalies_caught} of ${hybrid.anomalous_orders} anomalies; the rest look too much like normal orders.`}
                </div>
              </div>
            )}
            <ResultView record={result} />
          </>
        )}
      </div>
    </>
  );
}

function parseBatch(text: string): AnalyzeRequest[] {
  return text.split("\n").map((l) => l.trim()).filter((l) => l.length > 0).map((line) => {
    const m = line.match(/^([^:;]+):\s*(.*)$/);
    return m ? { order_id: m[1].trim(), event_sequence: m[2] } : { event_sequence: line };
  });
}

function Batch({ examples }: { examples: ExampleOrder[] }) {
  const exampleText = examples.map((o) => `${o.order_id}: ${o.event_sequence.join(";")}`).join("\n");
  const [text, setText] = useState(exampleText);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<Error | null>(null);
  const [res, setRes] = useState<BatchResponse | null>(null);
  const [open, setOpen] = useState<AnalysisRecord | null>(null);
  const items = parseBatch(text);

  const submit = async () => {
    setBusy(true);
    setError(null);
    try {
      setRes(await api.analyzeBatch(items));
    } catch (e) {
      setError(e as Error);
    } finally {
      setBusy(false);
    }
  };

  return (
    <>
      <section className="card">
        <div className="card-head">
          <div>
            <h3 className="card-title">One order per line</h3>
            <p className="card-sub">
              <span className="mono">ORDER-ID: EVENT;EVENT;…</span> or just the events. Up to 500 lines; all are stored together or not at all.
              Timing data can't be given here; use the single-order form for that.
            </p>
          </div>
          <button className="btn sm" onClick={() => setText(exampleText)}>Load the test examples</button>
        </div>
        <textarea className="textarea" value={text} onChange={(e) => setText(e.target.value)} spellCheck={false} aria-label="Batch input" />
        <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginTop: 12 }}>
          <span className="faint">{items.length} order{items.length === 1 ? "" : "s"}</span>
          <button className="btn primary" disabled={busy || items.length === 0 || items.length > 500} onClick={submit}>
            <Icon name={busy ? "loader" : "layers"} className={busy ? "spin" : undefined} />
            {busy ? "Analyzing…" : `Analyze ${items.length} orders`}
          </button>
        </div>
        {error !== null && <div style={{ marginTop: 12 }}><ErrorBox error={error} /></div>}
      </section>

      {res !== null && (
        <section className="card" style={{ marginTop: 16 }}>
          <div className="card-head">
            <div>
              <h3 className="card-title">{res.summary.total_orders} orders analyzed</h3>
              <p className="card-sub">Anomaly rate {(res.summary.anomaly_rate * 100).toFixed(1)}%. Click a row for details.</p>
            </div>
            <div className="pills">
              {VERDICT_ORDER.map((v) => {
                const n = res.results.filter((r) => r.verdict === v).length;
                return n > 0 ? <span key={v} className={`badge tone-${VERDICTS[v].tone}`}>{VERDICTS[v].label} · {n}</span> : null;
              })}
            </div>
          </div>
          <div className="table-wrap">
            <table>
              <thead><tr><th>Verdict</th><th>Order</th><th>Path</th><th>Why</th></tr></thead>
              <tbody>
                {res.results.map((r) => (
                  <tr key={r.id} className="clickable" onClick={() => setOpen(r)}>
                    <td><VerdictBadge verdict={r.verdict} /></td>
                    <td className="mono">{r.order_id === null ? "—" : r.order_id}</td>
                    <td><PathChips events={eventsOf(r)} max={5} failedAt={r.fsm_evaluation.error === null ? undefined : r.fsm_evaluation.error.failed_step} /></td>
                    <td className="muted" style={{ fontSize: 12.5 }}>{r.diagnostics.root_cause}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </section>
      )}

      {open !== null && (
        <Drawer title={`Analysis #${open.id}`} onClose={() => setOpen(null)}>
          <ResultView record={open} />
        </Drawer>
      )}
    </>
  );
}
