import type { AnalysisRecord, DetectorName, FsmDefinition, Severity, Verdict } from "../types";

export const EVENT_LABELS: Record<string, string> = {
  CONFIRM_PAYMENT: "Confirm payment",
  FAIL_PAYMENT: "Fail payment",
  RETRY_PAYMENT: "Retry payment",
  CANCEL_ORDER: "Cancel order",
  PACK_ITEMS: "Pack items",
  SHIP_ITEMS: "Ship items",
  DELIVER_ITEMS: "Deliver items",
  REFUND_ORDER: "Refund order",
};

export const EVENTS = Object.keys(EVENT_LABELS);

export const VERDICTS: Record<Verdict, { label: string; tone: string; severity: Severity; blurb: string }> = {
  NORMAL: { label: "Normal", tone: "ok", severity: "INFO", blurb: "Legal, finished, and nothing unusual." },
  RULE_VIOLATION: { label: "Rule violation", tone: "bad", severity: "CRITICAL", blurb: "A step that the workflow does not allow." },
  STATISTICAL_ANOMALY: {
    label: "Statistical anomaly", tone: "warn", severity: "WARNING",
    blurb: "Legal, but the event order, timing or amount is unusual.",
  },
  VALID_INCOMPLETE: { label: "In progress", tone: "info", severity: "INFO", blurb: "Legal so far, but the order has not finished." },
};

export const VERDICT_ORDER: Verdict[] = ["NORMAL", "STATISTICAL_ANOMALY", "RULE_VIOLATION", "VALID_INCOMPLETE"];

/** A UI label; a name the UI has no label for is an error, not a silent fallback. */
export function labelOf(labels: Record<string, string>, key: string): string {
  if (!(key in labels)) throw new Error(`[UI] expected a label for '${key}', got none; input=${Object.keys(labels).join(", ")}`);
  return labels[key];
}

export const DETECTOR_LABELS: Record<DetectorName, string> = {
  isolation_forest: "Isolation Forest",
  kmeans: "k-means clustering",
  local_outlier_factor: "Local Outlier Factor",
};

export const CHECK_LABELS: Record<string, string> = {
  path: "Event order",
  cluster: "Cluster",
  trend: "Trend",
  deviation: "Deviation",
};

export const CHECK_QUESTIONS: Record<string, string> = {
  path: "Is this order of events rare in normal orders?",
  cluster: "Are the times, amount and outcome far from typical normal orders?",
  trend: "Were the steps consistently faster, or consistently slower, than normal?",
  deviation: "Is any value far outside its normal range?",
};

export const MODEL_LABELS: Record<string, string> = {
  always_normal: "Always “normal”",
  range_rule: "Range rule, untuned (no ML)",
  tuned_range_rule: "Range rule, tuned (no ML)",
  zscore_or_markov: "z-score sum OR Markov chain",
  markov_chain: "Markov chain",
  timing_isolation_forest: "Isolation Forest",
  timing_kmeans: "k-means clustering",
  timing_local_outlier_factor: "Local Outlier Factor",
  trend_check: "Trend check",
  deviation_check: "Deviation check",
  hybrid: "Hybrid: four checks, one threshold",
  random_forest: "Random Forest",
};

export const MODEL_ROLES: Record<string, string> = {
  always_normal: "trivial baseline",
  range_rule: "non-ML baseline",
  tuned_range_rule: "non-ML baseline, same false-alarm budget",
  zscore_or_markov: "baseline without a fitted detector",
  markov_chain: "event order, alone",
  timing_isolation_forest: "cluster check candidate, alone",
  timing_kmeans: "cluster check candidate, alone",
  timing_local_outlier_factor: "cluster check candidate, alone",
  trend_check: "timing check, alone",
  deviation_check: "timing check, alone",
  hybrid: "used by the app",
  random_forest: "supervised comparison",
};

export const ANOMALY_TYPE_LABELS: Record<string, string> = {
  CARD_TESTING: "Card testing",
  REPEATED_PAYMENT_FAILURE: "Repeated failures",
  FAST_FULFILMENT: "Fast fulfilment",
  STALLED_DELIVERY: "Stalled delivery",
  HIGH_VALUE_QUICK_REFUND: "High-value quick refund",
};

export const VIOLATION_TYPE_LABELS: Record<string, string> = {
  MISSING_PAYMENT: "Shipped without payment",
  SKIPPED_STEP: "Skipped a step",
  WRONG_ORDER: "Steps in the wrong order",
  DUPLICATE_EVENT: "Event logged twice",
  EVENT_AFTER_END: "Event after the end",
  UNKNOWN_EVENT: "Unknown event",
};

export function eventsOf(record: AnalysisRecord): string[] {
  const raw = record.raw_input_payload;
  return (Array.isArray(raw) ? raw : raw.split(";")).map((e) => e.trim());
}

export interface WalkStep {
  step: number;
  from: string;
  event: string;
  to: string | null; // null when the step broke a rule
  ok: boolean;
}

/** Replays events through the FSM, exactly like backend/fsm/validator.py, stopping at the first failure. */
export function walkPath(events: string[], fsm: FsmDefinition): WalkStep[] {
  const steps: WalkStep[] = [];
  let state = fsm.initial_state;
  for (let i = 0; i < events.length; i++) {
    const t = fsm.terminal_states.includes(state)
      ? undefined
      : fsm.transitions.find((x) => x.from_state === state && x.event === events[i]);
    steps.push({ step: i + 1, from: state, event: events[i], to: t ? t.to_state : null, ok: Boolean(t) });
    if (!t) break;
    state = t.to_state;
  }
  return steps;
}

/** Terminal states still reachable from `state`, with the shortest event sequence to each (BFS). */
export function possibleOutcomes(state: string, fsm: FsmDefinition): Record<string, string[]> {
  const paths: Record<string, string[]> = { [state]: [] };
  const queue = [state];
  while (queue.length) {
    const s = queue.shift() as string;
    fsm.transitions
      .filter((t) => t.from_state === s)
      .sort((a, b) => a.event.localeCompare(b.event))
      .forEach((t) => {
        if (!(t.to_state in paths)) {
          paths[t.to_state] = [...paths[s], t.event];
          queue.push(t.to_state);
        }
      });
  }
  return Object.fromEntries(fsm.terminal_states.filter((t) => t in paths).map((t) => [t, paths[t]]));
}
