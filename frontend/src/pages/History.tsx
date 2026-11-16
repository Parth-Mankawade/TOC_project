import { useState } from "react";
import { useSearchParams } from "react-router-dom";
import { api } from "../api";
import { Icon } from "../components/Icon";
import { ResultView } from "../components/ResultView";
import { Drawer, ErrorBox, Loading, PageHeader, PathChips, SeverityBadge, VerdictBadge } from "../components/ui";
import { formatDateTime, formatNumber } from "../lib/format";
import { useAsync } from "../lib/hooks";
import { VERDICTS, VERDICT_ORDER, eventsOf } from "../lib/workflow";
import type { AnalysisRecord, Verdict } from "../types";

const PAGE_SIZE = 15;

function Score({ check }: { check: { anomaly_score: number; flagged: boolean } | null }) {
  if (check === null) return <span className="faint" title="This check did not run">—</span>;
  return <span style={check.flagged ? { color: "var(--warn)", fontWeight: 600 } : undefined}>{check.anomaly_score.toFixed(2)}</span>;
}

function Rows({ items, onOpen, selected }: { items: AnalysisRecord[]; onOpen: (r: AnalysisRecord) => void; selected: number | null }) {
  return (
    <div className="table-wrap">
      <table>
        <thead>
          <tr>
            <th>#</th><th>Verdict</th><th>Order</th><th>Path</th>
            <th className="num" title="Markov chain: above 0 means flagged">Path score</th>
            <th className="num" title="Timing-and-amount model: above 0 means flagged">Timing score</th>
            <th>Severity</th><th className="num">Analyzed</th>
          </tr>
        </thead>
        <tbody>
          {items.map((r) => (
            <tr key={r.id} className={`clickable${selected === r.id ? " highlight" : ""}`} onClick={() => onOpen(r)}>
              <td className="faint num">{r.id}</td>
              <td><VerdictBadge verdict={r.verdict} /></td>
              <td className="mono">{r.order_id === null ? "—" : r.order_id}</td>
              <td><PathChips events={eventsOf(r)} max={4} failedAt={r.fsm_evaluation.error === null ? undefined : r.fsm_evaluation.error.failed_step} /></td>
              <td className="num"><Score check={r.ml_evaluation === null ? null : { anomaly_score: r.ml_evaluation.anomaly_score, flagged: r.ml_evaluation.is_ml_anomaly }} /></td>
              <td className="num"><Score check={r.timing_evaluation === null ? null : { anomaly_score: r.timing_evaluation.anomaly_score, flagged: r.timing_evaluation.is_timing_anomaly }} /></td>
              <td><SeverityBadge severity={r.diagnostics.severity} /></td>
              <td className="num faint" style={{ whiteSpace: "nowrap" }}>{formatDateTime(r.created_at)}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

export function History() {
  const [params, setParams] = useSearchParams();
  const verdictParam = params.get("verdict");
  const verdict = verdictParam === null ? undefined : (verdictParam as Verdict); // an invalid value comes back as a 422 from the API
  const anomaly = params.get("anomaly");
  const pageParam = params.get("page");
  const page = pageParam === null ? 0 : Number(pageParam);
  const orderParam = params.get("order");
  const orderQuery = orderParam === null ? "" : orderParam;
  const [search, setSearch] = useState(orderQuery);
  const [open, setOpen] = useState<AnalysisRecord | null>(null);

  const update = (changes: Record<string, string | null>) => {
    const next = new URLSearchParams(params);
    for (const [k, v] of Object.entries(changes)) {
      if (v === null) next.delete(k);
      else next.set(k, v);
    }
    if (!("page" in changes)) next.delete("page");
    setParams(next);
  };

  const list = useAsync(
    () => api.list({ skip: page * PAGE_SIZE, limit: PAGE_SIZE, verdict, is_anomaly: anomaly === null ? undefined : anomaly === "true" }),
    [verdict, anomaly, page],
  );
  const history = useAsync(() => (orderQuery ? api.orderHistory(orderQuery) : Promise.resolve(null)), [orderQuery]);
  const pages = list.data === null ? 1 : Math.max(1, Math.ceil(list.data.total / PAGE_SIZE));

  return (
    <div className="page">
      <PageHeader
        eyebrow="History"
        title={<>Every <em>analysis</em>, stored</>}
        description="Filter by verdict, look up all analyses of one order, and open any row for the full explanation."
      />

      <section className="card flat" style={{ padding: 14, marginBottom: 16 }}>
        <div style={{ display: "flex", gap: 12, flexWrap: "wrap", alignItems: "center", justifyContent: "space-between" }}>
          <div className="pills" role="group" aria-label="Filter by verdict">
            <button className={`pill${verdict === undefined ? " active" : ""}`} onClick={() => update({ verdict: null })}>All</button>
            {VERDICT_ORDER.map((v) => (
              <button key={v} className={`pill${verdict === v ? " active" : ""}`} onClick={() => update({ verdict: v })}>
                <span className={`swatch sw-${VERDICTS[v].tone}`} /> {VERDICTS[v].label}
              </button>
            ))}
          </div>
          <div style={{ display: "flex", gap: 8, alignItems: "center" }}>
            <select className="select" style={{ width: 170, borderRadius: 999, height: 36 }} aria-label="Filter by anomaly status"
              value={anomaly === null ? "" : anomaly} onChange={(e) => update({ anomaly: e.target.value === "" ? null : e.target.value })}>
              <option value="">Any status</option>
              <option value="true">Flagged only</option>
              <option value="false">Not flagged</option>
            </select>
            <form className="search" style={{ width: 240 }}
              onSubmit={(e) => { e.preventDefault(); update({ order: search.trim() === "" ? null : search.trim() }); }}>
              <Icon name="search" size={16} />
              <input className="input" style={{ height: 36 }} placeholder="Order ID, then Enter" value={search}
                onChange={(e) => setSearch(e.target.value)} aria-label="Search by order ID" />
            </form>
          </div>
        </div>
      </section>

      {orderQuery !== "" && (
        <section className="card" style={{ marginBottom: 16 }}>
          <div className="card-head">
            <div>
              <h3 className="card-title">Order <span className="mono">{orderQuery}</span></h3>
              <p className="card-sub">Every analysis stored for this order, newest first.</p>
            </div>
            <button className="btn sm" onClick={() => { setSearch(""); update({ order: null }); }}>
              <Icon name="x" size={14} /> Clear
            </button>
          </div>
          {history.loading && <Loading height={80} />}
          {history.error !== null && <ErrorBox error={history.error} onRetry={history.reload} />}
          {history.data !== null && history.data.returned_count === 0 && <p className="muted">No analyses stored for this order ID.</p>}
          {history.data !== null && history.data.returned_count > 0 && <Rows items={history.data.analyses} onOpen={setOpen} selected={null} />}
        </section>
      )}

      <section className="card">
        <div className="card-head">
          <div>
            <h3 className="card-title">{verdict === undefined ? "All analyses" : VERDICTS[verdict].label}</h3>
            <p className="card-sub">
              {list.data === null ? "Loading…" : `${formatNumber(list.data.total)} record${list.data.total === 1 ? "" : "s"} · newest first`}
            </p>
          </div>
          <div className="actions" style={{ alignItems: "center" }}>
            <span className="faint" style={{ fontSize: 12.5 }}>Page {page + 1} of {pages}</span>
            <button className="btn sm" disabled={page === 0} onClick={() => update({ page: String(page - 1) })} aria-label="Previous page">
              <Icon name="back" size={14} />
            </button>
            <button className="btn sm" disabled={page + 1 >= pages} onClick={() => update({ page: String(page + 1) })} aria-label="Next page">
              <Icon name="arrow" size={14} />
            </button>
          </div>
        </div>
        {list.error !== null && <ErrorBox error={list.error} onRetry={list.reload} />}
        {list.loading && list.data === null && <Loading height={300} />}
        {list.data !== null && list.data.items.length === 0 && (
          <div className="empty" style={{ padding: 34 }}>
            <h3>Nothing here</h3>
            <p>No analyses match these filters. Analyze an order, or run the demo stream on the Overview page.</p>
          </div>
        )}
        {list.data !== null && list.data.items.length > 0 && <Rows items={list.data.items} onOpen={setOpen} selected={open === null ? null : open.id} />}
      </section>

      {open !== null && (
        <Drawer title={`Analysis #${open.id}${open.order_id === null ? "" : ` · ${open.order_id}`}`} onClose={() => setOpen(null)}>
          <ResultView record={open} />
        </Drawer>
      )}
    </div>
  );
}
