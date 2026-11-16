// Shapes of the backend API responses (backend/schemas/). Kept in step with the Pydantic models.

export type Verdict = "NORMAL" | "RULE_VIOLATION" | "STATISTICAL_ANOMALY" | "VALID_INCOMPLETE";
export type Severity = "CRITICAL" | "WARNING" | "INFO";

export interface FsmError {
  error_type: "INVALID_TRANSITION" | "UNKNOWN_EVENT" | "TERMINAL_STATE_VIOLATION";
  failed_step: number;
  failed_event: string;
  state_at_failure: string;
  allowed_events: string[];
  message: string;
}

export interface FsmEvaluation {
  is_rule_valid: boolean;
  is_complete: boolean;
  final_state: string;
  total_steps: number;
  successful_transitions: number;
  error: FsmError | null;
}

export interface TransitionProbability {
  step: number;
  from_state: string;
  event: string;
  to_state: string;
  probability: number;
  is_unusual: boolean;
}

export interface MlEvaluation {
  prediction: "NORMAL" | "ANOMALY";
  is_ml_anomaly: boolean;
  anomaly_score: number;
  path_log10_probability: number;
  transitions: TransitionProbability[];
}

export type MeasurementUnit = "seconds" | "currency" | "count";

export interface Measurement {
  name: string;
  step: number | null;
  unit: MeasurementUnit;
  value: number;
  normal_low: number;
  normal_high: number;
  percentile: number;
  is_unusual: boolean;
}

export type DetectorName = "isolation_forest" | "kmeans" | "local_outlier_factor";

export interface TimingEvaluation {
  prediction: "NORMAL" | "ANOMALY";
  is_timing_anomaly: boolean;
  anomaly_score: number;
  model_name: DetectorName;
  inputs: Record<string, number>;
  measurements: Measurement[];
}

export interface Diagnostics {
  severity: Severity;
  root_cause: string;
  key_factors: string[];
  recommended_action: string;
}

export interface AnalysisRecord {
  id: number;
  order_id: string | null;
  input_format: "DELIMITED_STRING" | "EVENT_LIST";
  raw_input_payload: string | string[];
  verdict: Verdict;
  is_anomaly: boolean;
  anomaly_type: "RULE_VIOLATION" | "STATISTICAL_ANOMALY" | null;
  summary: string;
  fsm_evaluation: FsmEvaluation;
  event_timestamps: string[] | null;
  order_amount: number | null;
  ml_evaluation: MlEvaluation | null;
  ml_skip_reason: string | null;
  timing_evaluation: TimingEvaluation | null;
  timing_skip_reason: string | null;
  diagnostics: Diagnostics;
  created_at: string;
}

export interface Metrics {
  total_analyses: number;
  normal_count: number;
  rule_violation_count: number;
  statistical_anomaly_count: number;
  valid_incomplete_count: number;
  total_anomalies: number;
  anomaly_rate: number | null; // null when nothing is stored yet
  unique_orders_count: number;
}

export interface AnalysesPage {
  items: AnalysisRecord[];
  total: number;
  skip: number;
  limit: number;
}

export interface OrderHistory {
  order_id: string;
  returned_count: number;
  analyses: AnalysisRecord[];
}

export interface BatchSummary {
  total_orders: number;
  normal_count: number;
  rule_violation_count: number;
  statistical_anomaly_count: number;
  valid_incomplete_count: number;
  total_anomalies: number;
  anomaly_rate: number;
}

export interface BatchResponse {
  results: AnalysisRecord[];
  summary: BatchSummary;
}

export interface FsmTransition {
  from_state: string;
  event: string;
  to_state: string;
}

export interface Reachability {
  initial_state: string;
  reachable_states: string[];
  unreachable_states: string[];
  states_that_can_finish: string[];
  deadlock_states: string[];
  is_deadlock_free: boolean;
  shortest_paths: Record<string, string[]>;
  cycles: string[][];
  accepted_sequences_by_length: number[];
  language_is_infinite: boolean;
}

export interface FsmDefinition {
  states: string[];
  events: string[];
  initial_state: string;
  terminal_states: string[];
  transitions: FsmTransition[];
  reachability: Reachability;
}

// ---------------------------------------------------------------- models report

export interface MarkovTransition extends FsmTransition {
  count: number;
  probability: number;
}

export interface MarkovModelInfo {
  model_name: string;
  smoothing_alpha: number;
  training_sample_count: number;
  validation_sample_count: number;
  threshold_percentile: number;
  threshold_rule: string;
  threshold_log10_probability: number;
  transitions: MarkovTransition[];
}

export interface LogNormalStats {
  mean_ln: number;
  std_ln: number;
  count: number;
  median: number;
  normal_low: number;
  normal_high: number;
}

export interface TimingModelInfo {
  model_name: DetectorName;
  detector_params: Record<string, unknown>;
  model_inputs: string[];
  threshold: number;
  threshold_percentile: number;
  normal_band_percentiles: [number, number];
  training_sample_count: number;
  validation_sample_count: number;
  sklearn_version: string;
  amount: LogNormalStats;
  steps: Record<string, LogNormalStats>; // key "FROM_EVENT -> TO_EVENT"
  failure_counts: Record<string, number>;
}

export interface EvaluatedModel {
  model: string;
  f1: number;
  precision: number;
  recall: number;
  false_alarms: number;
  normal_orders: number;
  false_alarm_rate: number;
  anomalies_caught: number;
  anomalous_orders: number;
  roc_auc: number | null;
  caught_by_type: Record<string, { caught: number; total: number }>;
  trained_on_types?: string[];
  never_seen_types?: string[];
}

export interface CandidateResult {
  f1: number;
  precision: number;
  recall: number;
  false_alarm_rate: number;
  params: Record<string, unknown>;
}

export interface EvaluationReport {
  dataset: {
    orders_by_split: Record<string, Record<string, number>>;
    anomaly_types: string[];
    test_anomaly_share: number;
  };
  separability: {
    false_alarm_budget: number;
    trivial_if_recall_at_least: number;
    by_type: Record<string, { measurement: string; side: "high" | "low"; recall: number; anomalies: number }>;
  };
  model_selection: {
    split: string;
    metric: string;
    rule: string;
    threshold_percentile: number;
    normal_orders: number;
    anomalous_orders: number;
    candidates: Record<DetectorName, CandidateResult>;
    chosen: DetectorName;
  };
  headline: { metric: string; model: string; hybrid: number; always_normal: number; range_rule: number };
  test: { models: EvaluatedModel[] };
  rule_violations: {
    orders: number;
    rejected_by_fsm: number;
    by_type: Record<string, number>;
    by_fsm_error: Record<string, number>;
    note: string;
  };
}

export interface ModelsInfo {
  markov_chain: MarkovModelInfo;
  timing_model: TimingModelInfo;
  evaluation: EvaluationReport;
  unusual_step_probability: number;
}

// ---------------------------------------------------------------- health, examples, demo

export interface LoadedModel {
  path: string;
  model_name: string;
  training_sample_count: number;
}

export interface Readiness {
  service_status: "OPERATIONAL";
  database: "ok";
  markov_model: LoadedModel;
  timing_model: LoadedModel;
  supported_verdicts: string[];
}

export type OrderCategory = "NORMAL" | "STATISTICAL_ANOMALY" | "RULE_VIOLATION" | "IN_PROGRESS";

export interface ExampleOrder {
  order_id: string;
  category: OrderCategory;
  anomaly_type: string; // the generator's label, e.g. CARD_TESTING, SKIPPED_STEP, or NONE
  event_sequence: string[];
  event_timestamps: string[];
  order_amount: number;
}

export interface DemoOrders {
  orders: ExampleOrder[];
  composition: Record<string, number>;
}

export interface AnalyzeRequest {
  order_id?: string;
  event_sequence: string | string[];
  event_timestamps?: string[];
  order_amount?: number;
}
