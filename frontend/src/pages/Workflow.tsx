import { useState } from "react";
import { FsmDiagram } from "../components/FsmDiagram";
import { Icon } from "../components/Icon";
import { PathPlayer, type StepFacts } from "../components/PathPlayer";
import { ErrorBox, Loading, PageHeader } from "../components/ui";
import { useFsmState } from "../lib/hooks";
import { possibleOutcomes } from "../lib/workflow";

const NO_FACTS = new Map<number, StepFacts>(); // a shortest path is not an order the models scored

function EventPath({ events }: { events: string[] }) {
  if (!events.length) return <span className="faint">no events needed (start state)</span>;
  return (
    <span className="path">
      {events.map((e, i) => (
        <span key={i} className="path">{i > 0 && <span className="arrow">→</span>}<span className="chip">{e}</span></span>
      ))}
    </span>
  );
}

export function Workflow() {
  const fsmState = useFsmState();
  const [selected, setSelected] = useState<string>("PACKED");
  const [playing, setPlaying] = useState<string | null>(null);

  if (fsmState.error !== null) return <div className="page"><ErrorBox error={fsmState.error} onRetry={fsmState.reload} /></div>;
  if (fsmState.data === null) return <div className="page"><Loading height={420} /></div>;
  const fsm = fsmState.data;
  const r = fsm.reachability;
  if (!(selected in r.shortest_paths)) {
    throw new Error(`[UI] expected a shortest path to ${selected}, got none; input=reachability.shortest_paths`);
  }
  const outgoing = fsm.transitions.filter((t) => t.from_state === selected);
  const terminal = fsm.terminal_states.includes(selected);
  const maxCount = Math.max(...r.accepted_sequences_by_length, 1);

  return (
    <div className="page">
      <PageHeader
        eyebrow="Workflow & reachability"
        title={<>The order <em>automaton</em></>}
        description="The rules every order must follow, as a deterministic finite automaton, and what graph search proves about it."
      />

      <section className="card">
        <div className="card-head">
          <div>
            <h3 className="card-title">State diagram</h3>
            <p className="card-sub">Click a state to see how it is reached and where an order can go from there.</p>
          </div>
          <span className="badge tone-accent">{fsm.states.length} states · {fsm.transitions.length} transitions</span>
        </div>
        {playing === null ? (
          <FsmDiagram fsm={fsm} selected={selected} onSelect={(state) => { setSelected(state); setPlaying(null); }} />
        ) : (
          <>
            <PathPlayer fsm={fsm} events={r.shortest_paths[playing]} facts={NO_FACTS} autoPlay />
            <button className="btn sm" style={{ marginTop: 10 }} onClick={() => setPlaying(null)}>
              <Icon name="back" size={14} /> Back to the clickable diagram
            </button>
          </>
        )}
      </section>

      <div className="grid g-2" style={{ marginTop: 16, alignItems: "start" }}>
        <section className="card">
          <div className="eyebrow" style={{ marginBottom: 4 }}>Selected state</div>
          <h3 className="card-title mono" style={{ fontSize: 18 }}>{selected}</h3>
          <p className="card-sub">
            {terminal ? "Final (accepting) state: the order is finished and no further event is allowed."
              : selected === fsm.initial_state ? "Start state: every order begins here."
              : "Active state: the order is still in progress."}
          </p>
          <dl className="kv" style={{ marginTop: 16 }}>
            <dt>Shortest way here</dt>
            <dd>
              <EventPath events={r.shortest_paths[selected]} />
              {r.shortest_paths[selected].length > 0 && (
                <button className="btn sm" style={{ marginTop: 8 }} onClick={() => setPlaying(selected)}>
                  <Icon name="play" size={14} /> Play this path
                </button>
              )}
            </dd>
            <dt>Allowed next</dt>
            <dd className="path">
              {outgoing.length ? outgoing.map((t) => (
                <span key={t.event} className="chip" title={`→ ${t.to_state}`}>{t.event} → {t.to_state}</span>
              )) : <span className="faint">none</span>}
            </dd>
            <dt>Can still end in</dt>
            <dd>
              <ul className="list-plain" style={{ gap: 6 }}>
                {Object.entries(possibleOutcomes(selected, fsm)).map(([end, path]) => (
                  <li key={end} style={{ display: "flex", gap: 8, alignItems: "center", flexWrap: "wrap" }}>
                    <span className="mono" style={{ fontWeight: 600, fontSize: 12.5 }}>{end}</span>
                    {path.length > 0 && <><span className="faint">via</span><EventPath events={path} /></>}
                  </li>
                ))}
              </ul>
            </dd>
          </dl>
        </section>

        <section className="card tint">
          <h3 className="card-title">Formal definition</h3>
          <p className="card-sub" style={{ marginBottom: 14 }}>M = (Q, Σ, δ, q₀, F)</p>
          <dl className="kv">
            <dt>Q · states</dt>
            <dd className="path">{fsm.states.map((s) => <span key={s} className="chip">{s}</span>)}</dd>
            <dt>Σ · events</dt>
            <dd className="path">{fsm.events.map((e) => <span key={e} className="chip">{e}</span>)}</dd>
            <dt>δ · transitions</dt>
            <dd>{fsm.transitions.length} allowed (state, event) pairs; every other pair is a rule violation.</dd>
            <dt>q₀ · start</dt>
            <dd><span className="chip">{fsm.initial_state}</span></dd>
            <dt>F · final</dt>
            <dd className="path">{fsm.terminal_states.map((s) => <span key={s} className="chip">{s}</span>)}</dd>
          </dl>
        </section>
      </div>

      <h2 className="section-title">Reachability analysis</h2>
      <div className="grid g-3" style={{ gridTemplateColumns: "repeat(4, minmax(0, 1fr))" }}>
        <div className="stat">
          <span className="stat-label"><Icon name="workflow" size={15} /> Reachable states</span>
          <span className="stat-value">{r.reachable_states.length}/{fsm.states.length}</span>
          <span className="stat-note">{r.unreachable_states.length ? `unreachable: ${r.unreachable_states.join(", ")}` : "every state can occur"}</span>
        </div>
        <div className="stat">
          <span className="stat-label"><Icon name="shield" size={15} /> Deadlocks</span>
          <span className="stat-value" style={{ color: r.is_deadlock_free ? "var(--ok)" : "var(--bad)" }}>{r.deadlock_states.length}</span>
          <span className="stat-note">{r.is_deadlock_free ? "every order can always still finish" : r.deadlock_states.join(", ")}</span>
        </div>
        <div className="stat">
          <span className="stat-label"><Icon name="refresh" size={15} /> Cycles</span>
          <span className="stat-value">{r.cycles.length}</span>
          <span className="stat-note">{r.cycles.map((c) => c.join(" ↔ ")).join("; ") || "none"}</span>
        </div>
        <div className="stat">
          <span className="stat-label"><Icon name="layers" size={15} /> Language</span>
          <span className="stat-value">{r.language_is_infinite ? "∞" : "finite"}</span>
          <span className="stat-note">{r.language_is_infinite ? "infinitely many valid orders (regular)" : "finitely many valid orders"}</span>
        </div>
      </div>

      <div className="grid g-2" style={{ marginTop: 16, alignItems: "start" }}>
        <section className="card">
          <h3 className="card-title">Valid complete orders by length</h3>
          <p className="card-sub">
            How many different event sequences of each length end in a final state. The count never drops to zero because the
            payment retry loop can repeat, so the set of valid orders is infinite, yet still a regular language.
          </p>
          <div style={{ display: "flex", alignItems: "flex-end", gap: 8, height: 170, marginTop: 18 }}>
            {r.accepted_sequences_by_length.map((n, len) => (
              <div key={len} style={{ flex: 1, display: "flex", flexDirection: "column", alignItems: "center", gap: 6 }}>
                <span style={{ fontSize: 12, fontWeight: 600 }}>{n}</span>
                <div style={{ width: "100%", height: `${(n / maxCount) * 120}px`, minHeight: 2, borderRadius: 8,
                  background: n ? "var(--accent)" : "var(--surface-3)", opacity: n ? 0.35 + 0.65 * (n / maxCount) : 1 }} />
                <span className="faint" style={{ fontSize: 11.5 }}>{len}</span>
              </div>
            ))}
          </div>
          <p className="faint" style={{ margin: "8px 0 0", fontSize: 12, textAlign: "center" }}>sequence length (number of events)</p>
        </section>

        <section className="card">
          <h3 className="card-title">Shortest way to each state</h3>
          <p className="card-sub" style={{ marginBottom: 6 }}>Breadth-first search from {fsm.initial_state}.</p>
          <div className="table-wrap">
            <table>
              <tbody>
                {Object.entries(r.shortest_paths).map(([state, path]) => (
                  <tr key={state} className="clickable" onClick={() => setSelected(state)}>
                    <td className="mono" style={{ fontWeight: 600, whiteSpace: "nowrap" }}>{state}</td>
                    <td><EventPath events={path} /></td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </section>
      </div>

      <section className="card" style={{ marginTop: 16 }}>
        <h3 className="card-title">Transition function δ</h3>
        <p className="card-sub" style={{ marginBottom: 6 }}>The complete rule set. Any (state, event) pair not listed is rejected.</p>
        <div className="table-wrap">
          <table>
            <thead><tr><th>#</th><th>From state</th><th>Event</th><th>To state</th></tr></thead>
            <tbody>
              {fsm.transitions.map((t, i) => (
                <tr key={`${t.from_state}|${t.event}`}>
                  <td className="faint">{i + 1}</td>
                  <td className="mono">{t.from_state}</td>
                  <td><span className="chip">{t.event}</span></td>
                  <td className="mono">{t.to_state}{fsm.terminal_states.includes(t.to_state) && <span className="faint"> (final)</span>}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </section>
    </div>
  );
}
