// Only fetch (the network) is replaced; the code under test is the real api.ts.
import { afterEach, describe, expect, it, vi } from "vitest";
import { ApiError, request } from "./api";

function respond(status: number, body: string, contentType = "application/json") {
  vi.stubGlobal("fetch", vi.fn(async () => new Response(body, { status, headers: { "Content-Type": contentType } })));
}

async function failure(promise: Promise<unknown>): Promise<ApiError> {
  try {
    await promise;
  } catch (e) {
    expect(e).toBeInstanceOf(ApiError);
    return e as ApiError;
  }
  throw new Error("expected the request to fail");
}

afterEach(() => vi.unstubAllGlobals());

describe("request", () => {
  it("returns the body when every required key is there", async () => {
    respond(200, JSON.stringify({ total: 3, items: [] }));
    await expect(request("/api/v1/analyses", ["total", "items"])).resolves.toEqual({ total: 3, items: [] });
  });

  it("fails loudly when the response lacks a key the page needs", async () => {
    respond(200, JSON.stringify({ total: 3 }));
    const err = await failure(request("/api/v1/analyses", ["total", "items"]));
    expect(err.message).toBe('[UI] expected keys total, items in the response, got total; input=GET /api/v1/analyses');
    expect(err.checkpoint).toBe("UI");
  });

  it("passes the backend's error message, checkpoint and every detail through", async () => {
    respond(422, JSON.stringify({ error: { code: "INPUT_ERROR", checkpoint: "INPUT",
      message: "[INPUT] expected a, got b; input=c (and 1 more)", details: ["[INPUT] expected a, got b; input=c", "[INPUT] x"] } }));
    const err = await failure(request("/api/v1/analyses", ["id"], { method: "POST", body: "{}" }));
    expect(err).toMatchObject({ status: 422, checkpoint: "INPUT", message: "[INPUT] expected a, got b; input=c (and 1 more)" });
    expect(err.details).toEqual(["[INPUT] expected a, got b; input=c", "[INPUT] x"]);
  });

  it("an error without details has none", async () => {
    respond(503, JSON.stringify({ error: { code: "MODEL_LOAD_ERROR", checkpoint: "MODEL",
      message: "[MODEL] expected a saved model, got no file; input=x.json", details: null } }));
    const err = await failure(request("/api/v1/health/ready", []));
    expect(err).toMatchObject({ status: 503, checkpoint: "MODEL", details: [] });
  });

  it("a response that is not JSON is an error, with what came back", async () => {
    respond(502, "<html>Bad gateway</html>", "text/html");
    const err = await failure(request("/api/v1/metrics", []));
    expect(err.message).toContain("[UI] expected a JSON response, got 502 text/html: <html>Bad gateway</html>");
  });

  it("an error body that is not the backend's envelope is reported as such", async () => {
    respond(500, JSON.stringify({ detail: "boom" }));
    const err = await failure(request("/api/v1/metrics", []));
    expect(err.message).toContain('[UI] expected an error envelope {"error": {...}}, got {"detail":"boom"}');
  });

  it("no connection at all says so and keeps the browser's reason", async () => {
    vi.stubGlobal("fetch", vi.fn(async () => { throw new TypeError("Failed to fetch"); }));
    const err = await failure(request("/api/v1/metrics", []));
    expect(err.status).toBe(0);
    expect(err.message).toBe("[UI] expected a response from the backend, got no connection (Failed to fetch); input=GET /api/v1/metrics");
  });
});
