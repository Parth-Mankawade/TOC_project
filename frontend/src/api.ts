// Typed client for the backend REST API (UI checkpoint).
//
// Nothing is swallowed: no connection, a body that is not JSON, an error response, or a
// success response missing a key the page needs all throw ApiError, and the pages show
// its message. Backend errors keep their own "[CHECKPOINT] expected ..., got ...; input=..."
// message and every detail line. URLs are relative, so the same build works when served by
// FastAPI (/app/) and through the Vite dev-server proxy.
import type {
  AnalysesPage,
  AnalysisRecord,
  AnalyzeRequest,
  BatchResponse,
  BatchSummary,
  DemoOrders,
  ExampleOrder,
  FsmDefinition,
  Metrics,
  ModelsInfo,
  OrderHistory,
  Readiness,
  Verdict,
} from "./types";

export class ApiError extends Error {
  readonly status: number;
  readonly checkpoint: string;
  readonly details: string[];

  constructor(message: string, status: number, checkpoint: string, details: string[]) {
    super(message);
    this.name = "ApiError";
    this.status = status;
    this.checkpoint = checkpoint;
    this.details = details;
  }
}

interface ErrorEnvelope {
  error: { code: string; checkpoint: string; message: string; details: string[] | null };
}

function isEnvelope(body: unknown): body is ErrorEnvelope {
  if (typeof body !== "object" || body === null || !("error" in body)) return false;
  const e = (body as { error: unknown }).error;
  return typeof e === "object" && e !== null && typeof (e as ErrorEnvelope["error"]).message === "string"
    && typeof (e as ErrorEnvelope["error"]).checkpoint === "string";
}

const uiError = (expected: string, got: string, where: string, status: number) =>
  new ApiError(`[UI] expected ${expected}, got ${got}; input=${where}`, status, "UI", []);

/** Fetch `path`, parse JSON, and require `requiredKeys` in a successful response. */
export async function request<T>(path: string, requiredKeys: string[], init?: RequestInit): Promise<T> {
  const method = init === undefined || init.method === undefined ? "GET" : init.method; // fetch's own default
  const where = `${method} ${path}`;
  let res: Response;
  try {
    res = await fetch(path, { ...init, headers: { "Content-Type": "application/json" } });
  } catch (e) {
    throw uiError("a response from the backend", `no connection (${(e as Error).message})`, where, 0);
  }
  const text = await res.text();
  let body: unknown;
  try {
    body = JSON.parse(text);
  } catch {
    throw uiError("a JSON response", `${res.status} ${res.headers.get("content-type")}: ${text.slice(0, 120)}`, where, res.status);
  }
  if (!res.ok) {
    if (!isEnvelope(body)) {
      throw uiError('an error envelope {"error": {...}}', JSON.stringify(body).slice(0, 200), where, res.status);
    }
    const { message, checkpoint, details } = body.error;
    throw new ApiError(message, res.status, checkpoint, details === null ? [] : details);
  }
  const present = typeof body === "object" && body !== null ? Object.keys(body) : [];
  const missing = requiredKeys.filter((k) => !present.includes(k));
  if (missing.length) {
    throw uiError(`keys ${requiredKeys.join(", ")} in the response`, present.join(", "), where, res.status);
  }
  return body as T;
}

const post = <T>(path: string, keys: string[], body?: unknown) =>
  request<T>(path, keys, { method: "POST", body: body === undefined ? undefined : JSON.stringify(body) });

export interface ListParams {
  skip?: number;
  limit?: number;
  verdict?: Verdict;
  is_anomaly?: boolean;
}

const RECORD_KEYS = ["id", "verdict", "fsm_evaluation", "diagnostics", "raw_input_payload"];
const SUMMARY_KEYS = ["total_orders", "normal_count", "rule_violation_count", "statistical_anomaly_count",
  "valid_incomplete_count", "total_anomalies", "anomaly_rate"];

export const api = {
  ready: () => request<Readiness>("/api/v1/health/ready", ["service_status", "markov_model", "timing_model"]),
  metrics: () => request<Metrics>("/api/v1/metrics", ["total_analyses", "anomaly_rate", "unique_orders_count"]),
  list: (params: ListParams = {}) => {
    const q = new URLSearchParams();
    for (const [k, v] of Object.entries(params)) if (v !== undefined) q.set(k, String(v));
    return request<AnalysesPage>(`/api/v1/analyses?${q}`, ["items", "total", "skip", "limit"]);
  },
  get: (id: number) => request<AnalysisRecord>(`/api/v1/analyses/${id}`, RECORD_KEYS),
  orderHistory: (orderId: string) =>
    request<OrderHistory>(`/api/v1/orders/${encodeURIComponent(orderId)}/analyses`, ["order_id", "analyses"]),
  analyze: (body: AnalyzeRequest) => post<AnalysisRecord>("/api/v1/analyses", RECORD_KEYS, body),
  analyzeBatch: (traces: AnalyzeRequest[]) => post<BatchResponse>("/api/v1/analyses/batch", ["results", "summary"], { traces }),
  fsm: () => request<FsmDefinition>("/api/v1/fsm", ["states", "transitions", "initial_state", "terminal_states", "reachability"]),
  models: () => request<ModelsInfo>("/api/v1/models", ["markov_chain", "timing_model", "evaluation", "unusual_step_probability"]),
  examples: () => request<{ examples: ExampleOrder[] }>("/api/v1/examples", ["examples"]).then((r) => r.examples),
  demoOrders: () => request<DemoOrders>("/api/v1/demo", ["orders", "composition"]),
  loadDemo: () => post<BatchSummary>("/api/v1/demo", SUMMARY_KEYS),
};

/** The request body for analyzing one example order with all its data. */
export function exampleRequest(o: ExampleOrder): AnalyzeRequest {
  return { order_id: o.order_id, event_sequence: o.event_sequence, event_timestamps: o.event_timestamps, order_amount: o.order_amount };
}
