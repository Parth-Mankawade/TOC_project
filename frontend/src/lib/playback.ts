// Pure logic behind the demo animations: the frames of a path playback, and the stages of
// one analysis in the order the backend runs them. Every text comes from the record itself.
import type { AnalysisRecord } from "../types";
import { describeProbability, formatAmount } from "./format";
import { DETECTOR_LABELS, VERDICTS, type WalkStep } from "./workflow";

export interface Frame {
  index: number; // 0 = before the first event
  state: string; // where the order is after this frame (where it failed, for a broken step)
  step: WalkStep | null;
  done: boolean;
}

export function playbackFrames(walk: WalkStep[], initial: string): Frame[] {
  const frames: Frame[] = [{ index: 0, state: initial, step: null, done: walk.length === 0 }];
  walk.forEach((step, i) => {
    frames.push({ index: i + 1, state: step.ok && step.to !== null ? step.to : step.from, step, done: i === walk.length - 1 });
  });
  return frames;
}

export type StageKey = "input" | "rules" | "finished" | "path" | "timing" | "verdict";
export type StageStatus = "pass" | "fail" | "waiting" | "clear" | "flagged" | "skipped";

export interface Stage {
  key: StageKey;
  title: string;
  status: StageStatus;
  detail: string;
}

function required<T>(value: T | null, what: string): T {
  if (value === null) throw new Error(`[UI] expected ${what}, got null; input=analysis record`);
  return value;
}

export function analysisStages(record: AnalysisRecord): Stage[] {
  const fsm = record.fsm_evaluation;
  const n = fsm.total_steps;
  const ml = record.ml_evaluation;
  const timing = record.timing_evaluation;

  const input = record.order_amount !== null
    ? `${n} event(s), ${n} timestamp(s), amount ${formatAmount(record.order_amount)}`
    : `${n} event(s), no timestamps or amount`;

  let rules: Stage;
  if (fsm.is_rule_valid) {
    rules = { key: "rules", title: "Workflow rules (FSM)", status: "pass", detail: `${fsm.successful_transitions} step(s), every one allowed` };
  } else {
    const e = required(fsm.error, "an FSM error for an invalid order");
    const why = e.error_type === "UNKNOWN_EVENT" ? "is not an event of this workflow"
      : e.error_type === "TERMINAL_STATE_VIOLATION" ? `came after the order ended in ${e.state_at_failure}`
        : `is not allowed in ${e.state_at_failure}`;
    rules = { key: "rules", title: "Workflow rules (FSM)", status: "fail", detail: `step ${e.failed_step}: ${e.failed_event} ${why}` };
  }

  const finished: Stage = !fsm.is_rule_valid
    ? { key: "finished", title: "Finished?", status: "skipped", detail: "not checked: a rule was broken" }
    : fsm.is_complete
      ? { key: "finished", title: "Finished?", status: "pass", detail: `ended in ${fsm.final_state}` }
      : { key: "finished", title: "Finished?", status: "waiting", detail: `still in ${fsm.final_state}` };

  const path: Stage = ml === null
    ? { key: "path", title: "Event order (Markov chain)", status: "skipped", detail: required(record.ml_skip_reason, "a path skip reason") }
    : {
      key: "path", title: "Event order (Markov chain)", status: ml.is_ml_anomaly ? "flagged" : "clear",
      detail: `${describeProbability(ml.path_log10_probability)} · score ${ml.anomaly_score.toFixed(2)}`,
    };

  const timingStage: Stage = timing === null
    ? { key: "timing", title: "Timing & amount", status: "skipped", detail: required(record.timing_skip_reason, "a timing skip reason") }
    : {
      key: "timing", title: "Timing & amount", status: timing.is_timing_anomaly ? "flagged" : "clear",
      detail: `${DETECTOR_LABELS[timing.model_name]} · score ${timing.anomaly_score.toFixed(2)}`,
    };

  const verdictStatus: Record<AnalysisRecord["verdict"], StageStatus> = {
    NORMAL: "clear", STATISTICAL_ANOMALY: "flagged", RULE_VIOLATION: "fail", VALID_INCOMPLETE: "waiting",
  };

  return [
    { key: "input", title: "Input", status: "pass", detail: input },
    rules,
    finished,
    path,
    timingStage,
    { key: "verdict", title: "Verdict", status: verdictStatus[record.verdict], detail: VERDICTS[record.verdict].label },
  ];
}
