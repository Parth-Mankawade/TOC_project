import { useCallback, useEffect, useRef, useState } from "react";
import { Link, useNavigate } from "react-router-dom";
import { api, exampleRequest } from "../api";
import { FsmDiagram } from "../components/FsmDiagram";
import { Icon, type IconName } from "../components/Icon";
import { ResultView } from "../components/ResultView";
import { Donut, Drawer, ErrorBox, Loading, PageHeader, PathChips, VerdictBadge } from "../components/ui";
import { describeProbability, formatNumber, formatPercent, timeAgo } from "../lib/format";
import { type AsyncState, useAsync, useFsmState } from "../lib/hooks";
import { ANOMALY_TYPE_LABELS, DETECTOR_LABELS, VERDICTS, VERDICT_ORDER, VIOLATION_TYPE_LABELS, eventsOf, labelOf, walkPath } from "../lib/workflow";
import type { AnalysisRecord, DemoOrders, ExampleOrder, Metrics, OrderCategory, Verdict } from "../types";

const TONE_COLOR: Record<string, string> = { ok: "var(--ok)", warn: "#e3a008", bad: "var(--bad)", info: "#8e9bb0" };
const RATES = { "1 / s": 1000, "2 / s": 500, "4 / s": 250 } as const; // time per order in the live stream
type Rate = keyof typeof RATES;
const FEED_SIZE = 8;

const countOf = (m: Metrics, v: Verdict) => ({
  NORMAL: m.normal_count,
  STATISTICAL_ANOMALY: m.statistical_anomaly_count,
  RULE_VIOLATION: m.rule_violation_count,
  VALID_INCOMPLETE: m.valid_incomplete_count,
})[v];

/** The verdict a correct system gives an order of each generated category. */
const EXPECTED: Record<OrderCategory, Verdict> = {
  NORMAL: "NORMAL",
  STATISTICAL_ANOMALY: "STATISTICAL_ANOMALY",
  RULE_VIOLATION: "RULE_VIOLATION",
  IN_PROGRESS: "VALID_INCOMPLETE",
};

function generatedAs(o: ExampleOrder): string {
  if (o.category === "STATISTICAL_ANOMALY") return labelOf(ANOMALY_TYPE_LABELS, o.anomaly_type);
  if (o.category === "RULE_VIOLATION") return labelOf(VIOLATION_TYPE_LABELS, o.anomaly_type);
  return o.category === "IN_PROGRESS" ? "Unfinished order" : "Normal order";
}

interface Tally { caught: number; missed: number; falseAlarms: number; normalOk: number; rules: number; inProgress: number }
const EMPTY_TALLY: Tally = { caught: 0, missed: 0, falseAlarms: 0, normalOk: 0, rules: 0, inProgress: 0 };

function tallied(t: Tally, o: ExampleOrder, r: AnalysisRecord): Tally {
  const next = { ...t };
  if (o.category === "STATISTICAL_ANOMALY") {
    if (r.verdict === "STATISTICAL_ANOMALY") next.caught += 1;
    else next.missed += 1;
  } else if (o.category === "NORMAL") {
    if (r.verdict === "NORMAL") next.normalOk += 1;
    else next.falseAlarms += 1;
  } else if (o.category === "RULE_VIOLATION") {
    next.rules += 1;
  } else {
    next.inProgress += 1;
  }
  return next;
}

const sleep = (ms: number) => new Promise((resolve) => window.setTimeout(resolve, ms));

function LiveStream({ demo, onStored, onOpen }: {
  demo: AsyncState<DemoOrders>;
  onStored: () => void;
  onOpen: (r: AnalysisRecord) => void;
}) {
  const fsm = useFsmState();
  const [running, setRunning] = useState(false);
  const [index, setIndex] = useState(0);
  const [rate, setRate] = useState<Rate>("2 / s");
  const [feed, setFeed] = useState<{ order: ExampleOrder; record: AnalysisRecord }[]>([]);
  const [tally, setTally] = useState<Tally>(EMPTY_TALLY);
  const [error, setError] = useState<Error | null>(null);
  const indexRef = useRef(0);
  const rateRef = useRef<Rate>(rate);
  rateRef.current = rate;

  useEffect(() => {
    if (!running || demo.data === null) return;
    const orders = demo.data.orders;
    let stop = false;
    (async () => {
      while (!stop && indexRef.current < orders.length) {
        const t0 = Date.now();
        const order = orders[indexRef.current];
        let record: AnalysisRecord;
        try {
          record = await api.analyze(exampleRequest(order));
        } catch (e) {
          setError(e as Error);
          setRunning(false);
          return;
        }
        setFeed((f) => [{ order, record }, ...f].slice(0, FEED_SIZE));
        setTally((t) => tallied(t, order, record));
        indexRef.current += 1;
        setIndex(indexRef.current);
        const wait = RATES[rateRef.current] - (Date.now() - t0);
        if (wait > 0) await sleep(wait);
      }
      if (indexRef.current >= orders.length) {
        setRunning(false);
        onStored();
      }
    })();
    return () => {
      stop = true;
    };
  }, [running, demo.data, onStored]);

  const reset = () => {
    setRunning(false);
    indexRef.current = 0;
    setIndex(0);
    setFeed([]);
    setTally(EMPTY_TALLY);
    setError(null);
  };

  if (demo.error !== null) return <section className="card"><ErrorBox error={demo.error} onRetry={demo.reload} /></section>;
  if (demo.data === null) return <Loading height={300} />;
  const total = demo.data.orders.length;
  const latest = feed.length > 0 ? feed[0] : null;
  const anomalies = tally.caught + tally.missed;
  const normals = tally.normalOk + tally.falseAlarms;

  return (
    <section className="card stream">
      <div className="card-head">
        <div>
          <h3 className="card-title"><Icon name="stream" /> Live demo stream</h3>
          <p className="card-sub">
            Sends {total} real test orders ({Object.entries(demo.data.composition).map(([k, n]) => `${n} ${k.toLowerCase().replace("_", " ")}`).join(", ")})
            to the API one at a time, the way a shop would. Every result is stored. The counters compare each verdict with how the
            data generator made that order.
          </p>
        </div>
        <div className="actions" style={{ alignItems: "center" }}>
          <select className="select sm" value={rate} onChange={(e) => setRate(e.target.value as Rate)} aria-label="Orders per second">
            {(Object.keys(RATES) as Rate[]).map((r) => <option key={r}>{r}</option>)}
          </select>
          {running ? (
            <button className="btn" onClick={() => setRunning(false)}><Icon name="pause" /> Pause</button>
          ) : (
            <button className="btn primary" onClick={() => { if (index >= total) reset(); setError(null); setRunning(true); }}>
              <Icon name="play" /> {index === 0 ? "Start stream" : index >= total ? "Run again" : "Resume"}
            </button>
          )}
          {index > 0 && !running && <button className="btn ghost" onClick={reset}><Icon name="restart" /> Reset</button>}
        </div>
      </div>

      <div className="stream-progress"><span style={{ width: `${(index / total) * 100}%` }} /></div>
      <div className="faint" style={{ fontSize: 12, margin: "6px 0 12px" }}>{index} of {total} orders analyzed</div>
      {error !== null && <div style={{ marginBottom: 12 }}><ErrorBox error={error} /></div>}

      <div className="stream-grid">
        <div>
          {fsm.data !== null && latest !== null ? (
            <>
              <FsmDiagram
                key={latest.record.id}
                fsm={fsm.data}
                highlight={{
                  steps: walkPath(eventsOf(latest.record), fsm.data).map((s) => ({
                    ...s,
                    unusual: latest.record.ml_evaluation !== null && latest.record.ml_evaluation.transitions.some((t) => t.step === s.step && t.is_unusual),
                  })),
                  failure: latest.record.fsm_evaluation.error === null ? undefined
                    : { state: latest.record.fsm_evaluation.error.state_at_failure, event: latest.record.fsm_evaluation.error.failed_event },
                  current: latest.record.fsm_evaluation.final_state,
                }}
              />
              <p className="player-caption">
                <VerdictBadge verdict={latest.record.verdict} /> <span className="mono">{latest.order.order_id}</span> · {latest.record.summary}
              </p>
            </>
          ) : (
            <div className="empty" style={{ padding: 30 }}><p>Press <b>Start stream</b> to watch orders flow through the system.</p></div>
          )}
        </div>
        <div>
          <div className="tally">
            <TallyCell icon="alert" label="Anomalies caught" value={`${tally.caught} / ${anomalies}`} tone="warn" />
            <TallyCell icon="check" label="Normal, not flagged" value={`${tally.normalOk} / ${normals}`} tone="ok" />
            <TallyCell icon="info" label="False alarms" value={String(tally.falseAlarms)} tone="info" />
            <TallyCell icon="shield" label="Rule violations rejected" value={String(tally.rules)} tone="bad" />
          </div>
          <ul className="feed">
            {feed.map(({ order, record }) => (
              <li key={record.id} className="feed-row" onClick={() => onOpen(record)}>
                <VerdictBadge verdict={record.verdict} />
                <span className="mono feed-id">{order.order_id}</span>
                <span className={`feed-truth${record.verdict === EXPECTED[order.category] ? "" : " miss"}`}
                  title="How the data generator made this order">
                  {generatedAs(order)} {record.verdict === EXPECTED[order.category] ? "✓" : "✕"}
                </span>
              </li>
            ))}
          </ul>
        </div>
      </div>
    </section>
  );
}

function TallyCell({ icon, label, value, tone }: { icon: IconName; label: string; value: string; tone: string }) {
  return (
    <div className={`tally-cell tone-${tone}`}>
      <span className="stat-label"><Icon name={icon} size={14} /> {label}</span>
      <b>{value}</b>
    </div>
  );
}

function HowChecked() {
  const models = useAsync(() => api.models());
  if (models.error !== null) return <ErrorBox error={models.error} onRetry={models.reload} />;
  if (models.data === null) return <Loading height={160} />;
  const { markov_chain: markov, evaluation } = models.data;
  const steps: [IconName, string, string][] = [
    ["shield", "Rules", "Finite state machine: is every step allowed, and has the order finished?"],
    ["route", "Event order", `Markov chain: is the path rarer than ${describeProbability(markov.threshold_log10_probability)}?`],
    ["clock", "Timing & amount", `Three checks: ${DETECTOR_LABELS[evaluation.model_selection.chosen]} (picked on validation data) asks whether the times, amount and outcome are far from typical normal orders; a trend check asks whether the steps were consistently fast or slow; a deviation check asks whether any value is far outside its normal range.`],
    ["layers", "One threshold", `All four checks share one threshold, set so that ${formatPercent(evaluation.hybrid.validation_flag_rate)} of ${formatNumber(evaluation.hybrid.validation_normal_orders)} normal validation orders are flagged.`],
  ];
  return (
    <ol className="list-plain" style={{ marginTop: 14 }}>
      {steps.map(([icon, title, text], i) => (
        <li key={title} style={{ display: "flex", gap: 12, alignItems: "flex-start" }}>
          <span className="check-icon tone-accent" style={{ width: 30, height: 30 }}><Icon name={icon} size={16} /></span>
          <div style={{ fontSize: 13 }}>
            <strong>{i + 1}. {title}</strong>
            <div className="muted">{text}</div>
          </div>
        </li>
      ))}
    </ol>
  );
}

export function Overview() {
  const navigate = useNavigate();
  const metrics = useAsync(() => api.metrics());
  const recent = useAsync(() => api.list({ limit: 8 }));
  const demo = useAsync(() => api.demoOrders());
  const [loadingDemo, setLoadingDemo] = useState(false);
  const [demoError, setDemoError] = useState<Error | null>(null);
  const [open, setOpen] = useState<AnalysisRecord | null>(null);

  const reloadMetrics = metrics.reload;
  const reloadRecent = recent.reload;
  const onStored = useCallback(() => {
    reloadMetrics();
    reloadRecent();
  }, [reloadMetrics, reloadRecent]);

  const loadDemo = async () => {
    setLoadingDemo(true);
    setDemoError(null);
    try {
      await api.loadDemo();
      onStored();
    } catch (e) {
      setDemoError(e as Error);
    } finally {
      setLoadingDemo(false);
    }
  };

  const demoCount = demo.data === null ? "" : ` ${demo.data.orders.length}`;
  const demoButton = (
    <button className="btn" onClick={loadDemo} disabled={loadingDemo || demo.data === null}>
      <Icon name={loadingDemo ? "loader" : "database"} className={loadingDemo ? "spin" : undefined} />
      {loadingDemo ? "Analyzing…" : `Load${demoCount} demo orders at once`}
    </button>
  );

  const m = metrics.data;

  return (
    <div className="page">
      <PageHeader
        eyebrow="Overview"
        title={<>Order workflow <em>monitor</em></>}
        description="Every order is checked against the workflow rules first, then by two machine learning models: one for the order of events, one for timing and amount."
        actions={<>{demoButton}<Link className="btn primary" to="/analyze"><Icon name="analyze" /> Analyze an order</Link></>}
      />

      {demoError !== null && <div style={{ marginBottom: 16 }}><ErrorBox error={demoError} /></div>}

      <LiveStream demo={demo} onStored={onStored} onOpen={setOpen} />

      {metrics.error !== null && <div style={{ marginTop: 16 }}><ErrorBox error={metrics.error} onRetry={metrics.reload} /></div>}
      {metrics.loading && m === null && <div className="grid g-6" style={{ marginTop: 16 }}>{Array.from({ length: 6 }, (_, i) => <Loading key={i} height={118} />)}</div>}

      {m !== null && m.total_analyses === 0 && (
        <div className="empty" style={{ marginTop: 16 }}>
          <span className="orb" style={{ width: 46, height: 46 }} />
          <h3>No orders analyzed yet</h3>
          <p>Start the live stream above, load the demo orders at once, or analyze your own order.</p>
        </div>
      )}

      {m !== null && m.total_analyses > 0 && m.anomaly_rate !== null && (
        <>
          <div className="grid g-6" style={{ marginTop: 16 }}>
            <Link className="stat" to="/history">
              <span className="stat-label"><Icon name="layers" size={15} /> Analyzed</span>
              <span className="stat-value">{formatNumber(m.total_analyses)}</span>
              <span className="stat-note">{formatNumber(m.unique_orders_count)} distinct order IDs</span>
            </Link>
            {VERDICT_ORDER.map((v) => (
              <button key={v} className="stat" onClick={() => navigate(`/history?verdict=${v}`)}>
                <span className="stat-label"><span className={`swatch sw-${VERDICTS[v].tone}`} /> {VERDICTS[v].label}</span>
                <span className="stat-value">{formatNumber(countOf(m, v))}</span>
                <span className="stat-note">{formatPercent(countOf(m, v) / m.total_analyses)} of all</span>
              </button>
            ))}
            <div className="stat dark">
              <span className="stat-label"><Icon name="alert" size={15} /> Anomaly rate</span>
              <span className="stat-value">{formatPercent(m.anomaly_rate)}</span>
              <span className="stat-note">{formatNumber(m.total_anomalies)} flagged orders</span>
            </div>
          </div>

          <div className="grid g-3" style={{ marginTop: 16 }}>
            <section className="card">
              <h3 className="card-title">Verdicts</h3>
              <p className="card-sub">Share of every analysis stored so far.</p>
              <div style={{ display: "flex", alignItems: "center", gap: 20, marginTop: 14 }}>
                <div style={{ position: "relative", flex: "none" }}>
                  <Donut parts={VERDICT_ORDER.map((v) => ({ value: countOf(m, v), color: TONE_COLOR[VERDICTS[v].tone] }))} />
                  <div style={{ position: "absolute", inset: 0, display: "grid", placeItems: "center", textAlign: "center" }}>
                    <div>
                      <div style={{ fontFamily: "var(--font-serif)", fontSize: 30, lineHeight: 1 }}>{formatNumber(m.total_analyses)}</div>
                      <div className="faint" style={{ fontSize: 11.5 }}>orders</div>
                    </div>
                  </div>
                </div>
                <div className="legend" style={{ flex: 1 }}>
                  {VERDICT_ORDER.map((v) => (
                    <div className="legend-row" key={v}>
                      <span className={`swatch sw-${VERDICTS[v].tone}`} />
                      <span>{VERDICTS[v].label}</span>
                      <b>{formatNumber(countOf(m, v))}</b>
                    </div>
                  ))}
                </div>
              </div>
            </section>

            <section className="card">
              <h3 className="card-title">Severity</h3>
              <p className="card-sub">Critical = broke a rule · Warning = suspicious · Info = fine or in progress.</p>
              <div className="legend" style={{ marginTop: 18, gap: 16 }}>
                {[
                  { label: "Critical", value: m.rule_violation_count, tone: "bad" },
                  { label: "Warning", value: m.statistical_anomaly_count, tone: "warn" },
                  { label: "Info", value: m.normal_count + m.valid_incomplete_count, tone: "info" },
                ].map((s) => (
                  <div key={s.label}>
                    <div style={{ display: "flex", justifyContent: "space-between", fontSize: 13, marginBottom: 6 }}>
                      <span>{s.label}</span>
                      <b style={{ fontVariantNumeric: "tabular-nums" }}>{formatNumber(s.value)}</b>
                    </div>
                    <div className={`bar ${s.tone}`}><span style={{ width: `${(s.value / m.total_analyses) * 100}%` }} /></div>
                  </div>
                ))}
              </div>
            </section>

            <section className="card tint">
              <h3 className="card-title">How each order is checked</h3>
              <HowChecked />
              <Link to="/workflow" className="btn sm" style={{ marginTop: 16 }}>See the workflow <Icon name="arrow" size={14} /></Link>
            </section>
          </div>

          <section className="card" style={{ marginTop: 16 }}>
            <div className="card-head">
              <div>
                <h3 className="card-title">Recent analyses</h3>
                <p className="card-sub">Newest first. Click a row for the full explanation.</p>
              </div>
              <Link to="/history" className="btn sm">View all <Icon name="arrow" size={14} /></Link>
            </div>
            {recent.error !== null && <ErrorBox error={recent.error} onRetry={recent.reload} />}
            {recent.data !== null && (
              <div className="table-wrap">
                <table>
                  <thead><tr><th>Verdict</th><th>Order</th><th>Path</th><th>Why</th><th className="num">When</th></tr></thead>
                  <tbody>
                    {recent.data.items.map((r) => (
                      <tr key={r.id} className="clickable" onClick={() => setOpen(r)}>
                        <td><VerdictBadge verdict={r.verdict} /></td>
                        <td className="mono">{r.order_id === null ? "—" : r.order_id}</td>
                        <td><PathChips events={eventsOf(r)} max={4} failedAt={r.fsm_evaluation.error === null ? undefined : r.fsm_evaluation.error.failed_step} /></td>
                        <td className="muted" style={{ maxWidth: 360, fontSize: 12.5 }}>{r.diagnostics.root_cause}</td>
                        <td className="num faint" style={{ whiteSpace: "nowrap" }}>{timeAgo(r.created_at)}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            )}
          </section>
        </>
      )}

      {open !== null && (
        <Drawer title={`Analysis #${open.id}`} onClose={() => setOpen(null)}>
          <ResultView record={open} />
        </Drawer>
      )}
    </div>
  );
}
