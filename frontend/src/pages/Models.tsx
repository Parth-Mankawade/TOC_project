import { api } from "../api";
import { FsmDiagram } from "../components/FsmDiagram";
import { Icon } from "../components/Icon";
import { Bar, ErrorBox, Loading, PageHeader } from "../components/ui";
import { describeProbability, formatAmount, formatDuration, formatNumber, formatPercent } from "../lib/format";
import { useAsync, useFsmState } from "../lib/hooks";
import { ANOMALY_TYPE_LABELS, DETECTOR_LABELS, MODEL_LABELS, MODEL_ROLES, VIOLATION_TYPE_LABELS, labelOf } from "../lib/workflow";
import type { DetectorName, EvaluatedModel, EvaluationReport } from "../types";

const STRONG = 0.5; // a model "catches most" of a type at or above this share
const WEAK = 0.2;   // and "mostly misses" it below this share

function share(m: EvaluatedModel, type: string) {
  const c = m.caught_by_type[type];
  return c.caught / c.total;
}

function byName(report: EvaluationReport, name: string): EvaluatedModel {
  const m = report.test.models.find((x) => x.model === name);
  if (m === undefined) throw new Error(`[UI] expected model '${name}' in the evaluation report, got none; input=GET /api/v1/models`);
  return m;
}

function Caught({ caught, total }: { caught: number; total: number }) {
  const s = caught / total;
  return (
    <div style={{ minWidth: 70 }}>
      <div style={{ fontSize: 12.5, fontVariantNumeric: "tabular-nums", marginBottom: 4 }}>{caught}/{total}</div>
      <Bar value={s} tone={s >= STRONG ? "ok" : s >= WEAK ? "warn" : "bad"} />
    </div>
  );
}

/** Sentences built only from the numbers in the report. */
function observations(report: EvaluationReport): string[] {
  const types = report.dataset.anomaly_types;
  const names = (list: string[], m: EvaluatedModel) =>
    list.map((t) => `${labelOf(ANOMALY_TYPE_LABELS, t)} (${m.caught_by_type[t].caught}/${m.caught_by_type[t].total})`).join(", ");
  const out: string[] = [];
  const hybrid = byName(report, "hybrid");
  const rule = byName(report, "range_rule");
  out.push(`The app's model (Markov chain + ${DETECTOR_LABELS[report.model_selection.chosen]}) reaches F1 ${hybrid.f1.toFixed(3)}, `
    + `against ${rule.f1.toFixed(3)} for the no-ML range rule: ${hybrid.anomalies_caught} anomalies caught instead of `
    + `${rule.anomalies_caught}, with ${hybrid.false_alarms} false alarms instead of ${rule.false_alarms}.`);
  for (const name of ["markov_chain", `timing_${report.model_selection.chosen}`]) {
    const m = byName(report, name);
    const strong = types.filter((t) => share(m, t) >= STRONG);
    const weak = types.filter((t) => share(m, t) < WEAK);
    out.push(`${labelOf(MODEL_LABELS, name)} catches most of: ${strong.length ? names(strong, m) : "no type"}; `
      + `mostly misses: ${weak.length ? names(weak, m) : "no type"}.`);
  }
  const rf = byName(report, "random_forest");
  if (rf.never_seen_types !== undefined && rf.trained_on_types !== undefined) {
    out.push(`The supervised Random Forest has the best F1 (${rf.f1.toFixed(3)}) because it was shown labelled examples of `
      + `${rf.trained_on_types.map((t) => labelOf(ANOMALY_TYPE_LABELS, t)).join(", ")}. On the types it never saw it caught `
      + `${names(rf.never_seen_types, rf)}: a real system rarely has labels for every new kind of anomaly.`);
  }
  return out;
}

export function Models() {
  const fsm = useFsmState();
  const info = useAsync(() => api.models());

  if (info.error !== null) return <div className="page"><ErrorBox error={info.error} onRetry={info.reload} /></div>;
  if (info.data === null) return <div className="page"><Loading height={420} /></div>;

  const { markov_chain: markov, timing_model: timing, evaluation: report, unusual_step_probability: unusualP } = info.data;
  const types = report.dataset.anomaly_types;
  const test = report.test.models[0];
  const hybrid = byName(report, "hybrid");
  const probabilities = Object.fromEntries(markov.transitions.map((t) => [`${t.from_state}|${t.event}`, t.probability]));
  const selection = report.model_selection;
  const [bandLow, bandHigh] = timing.normal_band_percentiles;

  return (
    <div className="page">
      <PageHeader
        eyebrow="Models & results"
        title={<>What the models <em>learned</em></>}
        description="The models learn from normal training orders only. Thresholds and the choice of timing model use a separate validation split. Every number below is measured once, on test orders no model saw."
      />

      <section className="card headline">
        <div>
          <div className="eyebrow">Headline · F1 on the held-out test split</div>
          <p className="card-sub" style={{ marginTop: 6 }}>
            {formatNumber(test.normal_orders)} normal orders and {formatNumber(test.anomalous_orders)} anomalies
            ({formatPercent(report.dataset.test_anomaly_share)} of the test split). F1 is high only when most anomalies are caught
            and most alarms are real.
          </p>
        </div>
        <div className="headline-grid">
          {[
            { label: "Markov + " + DETECTOR_LABELS[selection.chosen], value: report.headline.hybrid, note: "what the app uses", tone: "accent" },
            { label: "Range rule, no ML", value: report.headline.range_rule, note: "flag anything outside the normal min–max", tone: "info" },
            { label: "Always “normal”", value: report.headline.always_normal, note: "trivial baseline", tone: "info" },
          ].map((h) => (
            <div key={h.label} className={`stat tone-${h.tone}`}>
              <span className="stat-label">{h.label}</span>
              <span className="stat-value">{h.value.toFixed(3)}</span>
              <span className="stat-note">{h.note}</span>
            </div>
          ))}
        </div>
      </section>

      <section className="card" style={{ marginTop: 16 }}>
        <div className="card-head">
          <div>
            <h3 className="card-title">Every model on the test split</h3>
            <p className="card-sub">Cells show how many of each anomaly type were caught (of {test.caught_by_type[types[0]].total} each).</p>
          </div>
        </div>
        <div className="table-wrap">
          <table>
            <thead>
              <tr>
                <th>Model</th>
                <th className="num">False alarms</th>
                {types.map((t) => <th key={t}>{labelOf(ANOMALY_TYPE_LABELS, t)}</th>)}
                <th className="num">Precision</th>
                <th className="num">Recall</th>
                <th className="num">F1</th>
                <th className="num">ROC-AUC</th>
              </tr>
            </thead>
            <tbody>
              {report.test.models.map((m) => (
                <tr key={m.model} className={m.model === "hybrid" ? "highlight" : undefined}>
                  <td>
                    <div style={{ fontWeight: 600 }}>{labelOf(MODEL_LABELS, m.model)}</div>
                    <span className="badge tone-info" style={{ marginTop: 4, height: 20, fontSize: 11 }}>{labelOf(MODEL_ROLES, m.model)}</span>
                  </td>
                  <td className="num">
                    <div style={{ fontWeight: 600 }}>{m.false_alarms}</div>
                    <div className="faint" style={{ fontSize: 11.5 }}>{formatPercent(m.false_alarm_rate)}</div>
                  </td>
                  {types.map((t) => <td key={t}><Caught caught={m.caught_by_type[t].caught} total={m.caught_by_type[t].total} /></td>)}
                  <td className="num">{m.precision.toFixed(3)}</td>
                  <td className="num">{m.recall.toFixed(3)}</td>
                  <td className="num" style={{ fontWeight: 600 }}>{m.f1.toFixed(3)}</td>
                  <td className="num">{m.roc_auc === null ? <span className="faint" title="This model gives no score to rank by">—</span> : m.roc_auc.toFixed(3)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        <ul className="list-plain" style={{ marginTop: 14 }}>
          {observations(report).map((o, i) => (
            <li key={i} className="factor"><span className="mark" /><span>{o}</span></li>
          ))}
        </ul>
      </section>

      <div className="grid g-2" style={{ marginTop: 16, alignItems: "start" }}>
        <section className="card">
          <h3 className="card-title">Choosing the timing model (validation split)</h3>
          <p className="card-sub" style={{ marginBottom: 8 }}>
            {selection.normal_orders} normal and {selection.anomalous_orders} anomalous validation orders. Each candidate flags the
            most unusual {formatPercent(1 - selection.threshold_percentile / 100, 0)} of normal validation orders. Rule: {selection.rule}.
          </p>
          <div className="table-wrap">
            <table>
              <thead><tr><th>Candidate</th><th className="num">Recall</th><th className="num">False alarms</th><th className="num">F1</th></tr></thead>
              <tbody>
                {(Object.keys(selection.candidates) as DetectorName[]).map((name) => {
                  const c = selection.candidates[name];
                  return (
                    <tr key={name} className={name === selection.chosen ? "highlight" : undefined}>
                      <td>{DETECTOR_LABELS[name]}{name === selection.chosen && <span className="badge tone-accent" style={{ marginLeft: 8 }}>chosen</span>}</td>
                      <td className="num">{c.recall.toFixed(3)}</td>
                      <td className="num">{formatPercent(c.false_alarm_rate)}</td>
                      <td className="num" style={{ fontWeight: 600 }}>{c.f1.toFixed(3)}</td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        </section>

        <section className="card">
          <h3 className="card-title">Data check: no giveaway feature</h3>
          <p className="card-sub" style={{ marginBottom: 8 }}>
            For each anomaly type, the single measurement that catches the most of it while flagging at most
            {" "}{formatPercent(report.separability.false_alarm_budget, 0)} of normal orders. A type would be too easy (and ML pointless)
            at {formatPercent(report.separability.trivial_if_recall_at_least, 0)} or more.
          </p>
          <div className="table-wrap">
            <table>
              <thead><tr><th>Type</th><th>Best single measurement</th><th className="num">Catches</th></tr></thead>
              <tbody>
                {types.map((t) => {
                  const s = report.separability.by_type[t];
                  return (
                    <tr key={t}>
                      <td>{labelOf(ANOMALY_TYPE_LABELS, t)}</td>
                      <td className="mono" style={{ fontSize: 12 }}>{s.measurement} ({s.side})</td>
                      <td className="num">{formatPercent(s.recall, 0)}</td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        </section>
      </div>

      <section className="card tint" style={{ marginTop: 16, display: "flex", gap: 24, alignItems: "center", flexWrap: "wrap" }}>
        <div>
          <div className="stat-label"><Icon name="shield" size={15} /> Rule violations in the test split</div>
          <div className="stat-value" style={{ marginTop: 6 }}>
            {formatNumber(report.rule_violations.rejected_by_fsm)}<span className="faint" style={{ fontSize: 22 }}> / {formatNumber(report.rule_violations.orders)}</span>
          </div>
          <div className="faint" style={{ fontSize: 12, maxWidth: 420 }}>{report.rule_violations.note}</div>
        </div>
        <div className="pills">
          {Object.entries(report.rule_violations.by_type).map(([t, n]) => (
            <span key={t} className="badge tone-info">{labelOf(VIOLATION_TYPE_LABELS, t)} · {n}</span>
          ))}
        </div>
      </section>

      <h2 className="section-title">Markov chain: event order</h2>
      <section className="card">
        <div className="card-head">
          <div>
            <h3 className="card-title">Learned transition probabilities</h3>
            <p className="card-sub">
              P(event | state), counted in {formatNumber(markov.training_sample_count)} normal training orders (+{markov.smoothing_alpha} per
              transition so unseen ones are not impossible). Thicker arrows are more likely. An order is flagged when its whole path is
              rarer than the rarest {markov.threshold_percentile}% of {formatNumber(markov.validation_sample_count)} normal validation
              orders: {describeProbability(markov.threshold_log10_probability)}.
            </p>
          </div>
        </div>
        {fsm.error !== null ? <ErrorBox error={fsm.error} onRetry={fsm.reload} />
          : fsm.data === null ? <Loading height={260} /> : <FsmDiagram fsm={fsm.data} probabilities={probabilities} />}
        <div className="table-wrap" style={{ marginTop: 10 }}>
          <table>
            <thead><tr><th>From state</th><th>Event</th><th className="num">Seen</th><th style={{ width: "34%" }}>Probability</th></tr></thead>
            <tbody>
              {markov.transitions.map((t) => (
                <tr key={`${t.from_state}|${t.event}`}>
                  <td className="mono">{t.from_state}</td>
                  <td><span className="chip">{t.event}</span></td>
                  <td className="num">{formatNumber(t.count)}</td>
                  <td>
                    <div className="prob"><Bar value={t.probability} tone={t.probability < unusualP ? "warn" : undefined} /><span>{t.probability.toFixed(3)}</span></div>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </section>

      <h2 className="section-title">{DETECTOR_LABELS[timing.model_name]}: timing & amount</h2>
      <div className="grid g-2" style={{ alignItems: "start" }}>
        <section className="card">
          <h3 className="card-title">Normal ranges from {formatNumber(timing.training_sample_count)} normal training orders</h3>
          <p className="card-sub" style={{ marginBottom: 6 }}>
            Typical value (median) and the middle {bandHigh - bandLow}% of normal orders ({bandLow}th to {bandHigh}th percentile).
          </p>
          <div className="table-wrap">
            <table>
              <thead><tr><th>Measurement</th><th className="num">Typical</th><th className="num">Normal range</th></tr></thead>
              <tbody>
                <tr>
                  <td>Order amount</td>
                  <td className="num">{formatAmount(timing.amount.median)}</td>
                  <td className="num">{formatAmount(timing.amount.normal_low)} – {formatAmount(timing.amount.normal_high)}</td>
                </tr>
                {Object.entries(timing.steps).map(([k, s]) => (
                  <tr key={k}>
                    <td className="mono" style={{ fontSize: 12 }}>{k}</td>
                    <td className="num">{formatDuration(s.median)}</td>
                    <td className="num" style={{ whiteSpace: "nowrap" }}>{formatDuration(s.normal_low)} – {formatDuration(s.normal_high)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          <div className="pills" style={{ marginTop: 10 }}>
            {Object.entries(timing.failure_counts).map(([k, n]) => (
              <span key={k} className="badge tone-info">{k} failed payment{k === "1" ? "" : "s"} · {formatPercent(n / timing.training_sample_count)}</span>
            ))}
          </div>
        </section>
        <section className="card tint">
          <h3 className="card-title">How it decides</h3>
          <ol className="list-plain" style={{ marginTop: 12, gap: 12 }}>
            {[
              `Each order becomes ${timing.model_inputs.length} numbers: ${timing.model_inputs.join(", ")}. A z-score is how many standard deviations a value is from normal (on a log scale).`,
              `${DETECTOR_LABELS[timing.model_name]} (${Object.entries(timing.detector_params).filter(([, v]) => typeof v !== "object").map(([k, v]) => `${k} = ${v}`).join(", ")}) learns where normal orders lie in that space.`,
              `An order is flagged when it is more unusual than ${timing.threshold_percentile}% of ${formatNumber(timing.validation_sample_count)} normal validation orders.`,
              `On the test split the combined model flagged ${formatPercent(hybrid.false_alarm_rate)} of normal orders and caught ${hybrid.anomalies_caught} of ${hybrid.anomalous_orders} anomalies.`,
            ].map((text, i) => (
              <li key={i} style={{ display: "flex", gap: 12 }}>
                <span className="step-num">{i + 1}</span>
                <span style={{ fontSize: 13.5 }}>{text}</span>
              </li>
            ))}
          </ol>
        </section>
      </div>

      <div className="foot-note">
        <span>Results are on synthetic data and do not guarantee real-world performance.</span>
        <span>Reproduce everything: <span className="mono">python -m backend.ml.pipeline</span></span>
      </div>
    </div>
  );
}
