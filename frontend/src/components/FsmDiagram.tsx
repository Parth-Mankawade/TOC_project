import type { FsmDefinition } from "../types";

// Hand-placed layout so no two transitions cross: the main fulfilment path runs along the
// top row; the payment-failure loop, cancellation and refund sit below it.
const POS: Record<string, { x: number; y: number }> = {
  ORDER_CREATED: { x: 190, y: 120 },
  PAID: { x: 430, y: 120 },
  PACKED: { x: 640, y: 120 },
  SHIPPED: { x: 850, y: 120 },
  DELIVERED: { x: 1070, y: 120 },
  PAYMENT_FAILED: { x: 330, y: 290 },
  CANCELLED: { x: 95, y: 290 },
  REFUNDED: { x: 535, y: 290 },
};

function pos(state: string) {
  if (!(state in POS)) throw new Error(`[UI] expected a diagram position for state '${state}', got none; input=GET /api/v1/fsm`);
  return POS[state];
}

// curve: how far the edge bends to the left of its direction of travel.
// label: offset of the event name on that side; anchor: which end of the text sits at that point.
// labelT / badgeT: where along the edge (0 = start, 1 = end) the label and the pill sit.
interface EdgeStyle {
  curve: number;
  label: number;
  anchor: "start" | "middle" | "end";
  labelT: number;
  badgeT: number;
}
const DEFAULT_STYLE: EdgeStyle = { curve: 0, label: 16, anchor: "middle", labelT: 0.5, badgeT: 0.5 }; // straight top-row edges
const styled = (s: Partial<EdgeStyle>): EdgeStyle => ({ ...DEFAULT_STYLE, ...s });
const EDGE_STYLE: Record<string, EdgeStyle> = {
  "ORDER_CREATED|FAIL_PAYMENT": styled({ curve: 50, label: 12, anchor: "start", badgeT: 0.72 }),
  "PAYMENT_FAILED|RETRY_PAYMENT": styled({ curve: 50, label: 12, anchor: "end", badgeT: 0.72 }),
  "ORDER_CREATED|CANCEL_ORDER": styled({ label: -10, anchor: "end", badgeT: 0.85 }),
  "PAYMENT_FAILED|CANCEL_ORDER": styled({}),
  "PAYMENT_FAILED|CONFIRM_PAYMENT": styled({ label: -10, anchor: "start", labelT: 0.3, badgeT: 0.45 }),
  "PAID|REFUND_ORDER": styled({ label: 10, anchor: "start", badgeT: 0.72 }),
  "PACKED|REFUND_ORDER": styled({ label: 10, anchor: "start", badgeT: 0.72 }),
};

type Pt = { x: number; y: number };

/** Point at parameter t on the quadratic curve p0 -> (ctrl) -> p1. */
function onCurve(p0: Pt, ctrl: Pt, p1: Pt, t: number): Pt {
  const a = (1 - t) * (1 - t);
  const b = 2 * t * (1 - t);
  const c = t * t;
  return { x: a * p0.x + b * ctrl.x + c * p1.x, y: a * p0.y + b * ctrl.y + c * p1.y };
}

const NODE_H = 42;
const nodeW = (state: string) => Math.max(92, state.length * 7.6 + 32);

function exitPoint(c: Pt, state: string, toward: Pt, pad = 0): Pt {
  const dx = toward.x - c.x;
  const dy = toward.y - c.y;
  const hw = nodeW(state) / 2 + pad;
  const hh = NODE_H / 2 + pad;
  const t = Math.min(hw / Math.abs(dx === 0 ? 1e-9 : dx), hh / Math.abs(dy === 0 ? 1e-9 : dy));
  return { x: c.x + dx * t, y: c.y + dy * t };
}

export interface PathHighlight {
  steps: { step: number; from: string; event: string; to: string | null; unusual: boolean }[];
  failure?: { state: string; event: string };
  current?: string;
}

/** A dot travelling along one edge; a new `key` restarts the animation. */
export interface Token {
  edge: string; // "FROM_STATE|EVENT"
  key: number;
  ms: number;
}

interface Props {
  fsm: FsmDefinition;
  highlight?: PathHighlight;
  probabilities?: Record<string, number>; // "FROM|EVENT" -> probability
  selected?: string | null;
  onSelect?: (state: string) => void;
  token?: Token;
}

export function FsmDiagram({ fsm, highlight, probabilities, selected, onSelect, token }: Props) {
  const onEdges = new Map<string, { steps: number[]; unusual: boolean }>();
  const visited = new Set<string>();
  if (highlight) {
    visited.add(fsm.initial_state);
    for (const s of highlight.steps) {
      if (s.to === null) continue;
      const key = `${s.from}|${s.event}`;
      const entry = onEdges.has(key) ? onEdges.get(key)! : { steps: [], unusual: false };
      entry.steps.push(s.step);
      entry.unusual = entry.unusual || s.unusual;
      onEdges.set(key, entry);
      visited.add(s.from).add(s.to);
    }
  }
  const start = pos(fsm.initial_state);

  return (
    <svg className="fsm" viewBox="0 40 1150 320" role="img" aria-label="Order workflow state machine">
      <defs>
        {[["arrow", "#c3c0b8"], ["arrow-on", "var(--accent)"], ["arrow-unusual", "#e3a008"], ["arrow-dim", "#e2e0da"]].map(([id, color]) => (
          // userSpaceOnUse: arrowheads keep their size when a line is drawn thicker.
          <marker key={id} id={id} viewBox="0 0 10 10" refX="9" refY="5" markerWidth="10" markerHeight="10"
            markerUnits="userSpaceOnUse" orient="auto-start-reverse">
            <path d="M 0 0 L 10 5 L 0 10 z" fill={color} />
          </marker>
        ))}
      </defs>

      <g className="start">
        <line x1={start.x - 150} y1={start.y} x2={start.x - nodeW(fsm.initial_state) / 2 - 2} y2={start.y}
          stroke="#c3c0b8" strokeWidth="1.4" markerEnd="url(#arrow)" />
        <text x={start.x - 150} y={start.y - 10}>start</text>
      </g>

      {fsm.transitions.map((t) => {
        const key = `${t.from_state}|${t.event}`;
        const a = pos(t.from_state);
        const b = pos(t.to_state);
        const style = key in EDGE_STYLE ? EDGE_STYLE[key] : DEFAULT_STYLE;
        const dx = b.x - a.x;
        const dy = b.y - a.y;
        const len = Math.hypot(dx, dy);
        const n = { x: dy / len, y: -dx / len }; // left of the direction of travel (screen coordinates)
        const ctrl = { x: (a.x + b.x) / 2 + n.x * style.curve, y: (a.y + b.y) / 2 + n.y * style.curve };
        const p0 = exitPoint(a, t.from_state, style.curve ? ctrl : b);
        const p1 = exitPoint(b, t.to_state, style.curve ? ctrl : a, 2);
        const labelAt = onCurve(p0, ctrl, p1, style.labelT);
        const pill = onCurve(p0, ctrl, p1, style.badgeT);
        const d = `M ${p0.x} ${p0.y} Q ${ctrl.x} ${ctrl.y} ${p1.x} ${p1.y}`;

        const on = onEdges.get(key);
        const prob = probabilities !== undefined && key in probabilities ? probabilities[key] : null;
        const cls = on ? (on.unusual ? "edge on unusual" : "edge on") : highlight ? "edge dim" : "edge";
        const marker = on ? (on.unusual ? "arrow-unusual" : "arrow-on") : highlight ? "arrow-dim" : "arrow";
        const badgeText = on ? (on.steps.length > 3 ? `${on.steps.slice(0, 3).join(",")}…` : on.steps.join(",")) : "";
        const pillW = (text: string) => text.length * 6.6 + 14;

        return (
          <g key={key} className={cls}>
            <path d={d} markerEnd={`url(#${marker})`}
              style={prob === null ? undefined : { strokeWidth: 1 + prob * 3, stroke: "var(--accent)", strokeOpacity: 0.3 + prob * 0.7 }} />
            <text x={labelAt.x + n.x * style.label} y={labelAt.y + n.y * style.label + 3.5} textAnchor={style.anchor}>{t.event}</text>
            {on && (
              <g className={`step-badge${on.unusual ? " unusual" : ""}`}>
                <rect x={pill.x - pillW(badgeText) / 2} y={pill.y - 9} width={pillW(badgeText)} height={18} rx={9}
                  fill={on.unusual ? "#e3a008" : "var(--accent)"} />
                <text x={pill.x} y={pill.y + 3.5} textAnchor="middle">{badgeText}</text>
              </g>
            )}
            {prob !== null && (
              <g className="prob-pill">
                <rect x={pill.x - pillW("0.000") / 2} y={pill.y - 9} width={pillW("0.000")} height={18} rx={9} />
                <text x={pill.x} y={pill.y + 3.5} textAnchor="middle">{prob.toFixed(3)}</text>
              </g>
            )}
            {token && token.edge === key && (
              <circle key={token.key} r={7} className="travel">
                <animateMotion dur={`${token.ms}ms`} fill="freeze" path={d} />
              </circle>
            )}
          </g>
        );
      })}

      {fsm.states.map((s) => {
        const p = pos(s);
        const w = nodeW(s);
        const terminal = fsm.terminal_states.includes(s);
        const failed = highlight !== undefined && highlight.failure !== undefined && highlight.failure.state === s;
        const classes = [
          "node",
          terminal && "terminal",
          visited.has(s) && "visited",
          highlight !== undefined && highlight.current === s && !failed && "current",
          failed && "failed",
          selected === s && "selected",
          onSelect && "clickable",
        ].filter(Boolean).join(" ");
        return (
          <g key={s} className={classes} onClick={onSelect ? () => onSelect(s) : undefined}
            role={onSelect ? "button" : undefined} tabIndex={onSelect ? 0 : undefined}
            onKeyDown={onSelect ? (e) => (e.key === "Enter" || e.key === " ") && onSelect(s) : undefined}>
            <rect className="outer" x={p.x - w / 2} y={p.y - NODE_H / 2} width={w} height={NODE_H} rx={13} />
            {terminal && (
              <rect className="inner" x={p.x - w / 2 + 4} y={p.y - NODE_H / 2 + 4} width={w - 8} height={NODE_H - 8} rx={10}
                fill="none" stroke="var(--line-strong)" strokeWidth={1} />
            )}
            <text x={p.x} y={p.y + 4} textAnchor="middle">{s}</text>
          </g>
        );
      })}

      {highlight !== undefined && highlight.failure !== undefined && (() => {
        const p = pos(highlight.failure.state);
        const label = `✕ ${highlight.failure.event}`;
        const w = label.length * 7 + 18;
        const below = p.y > 200;
        const y = below ? p.y + NODE_H / 2 + 12 : p.y - NODE_H / 2 - 12 - 22;
        return (
          <g className="fail-flag">
            <line x1={p.x} y1={below ? p.y + NODE_H / 2 : p.y - NODE_H / 2} x2={p.x} y2={below ? y : y + 22}
              stroke="var(--bad)" strokeWidth={1.6} strokeDasharray="3 3" />
            <rect x={p.x - w / 2} y={y} width={w} height={22} rx={11} />
            <text x={p.x} y={y + 15} textAnchor="middle">{label}</text>
          </g>
        );
      })()}
    </svg>
  );
}
