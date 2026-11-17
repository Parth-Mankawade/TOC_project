import { describe, expect, it } from "vitest";
import { FSM, HAPPY, RECORDS } from "../test/fixtures";
import { analysisStages, playbackFrames } from "./playback";
import { walkPath } from "./workflow";

describe("playbackFrames: one frame before the first event, then one per step", () => {
  it("moves through a valid order and ends in its final state", () => {
    const frames = playbackFrames(walkPath(HAPPY, FSM), FSM.initial_state);
    expect(frames.map((f) => f.state)).toEqual(["ORDER_CREATED", "PAID", "PACKED", "SHIPPED", "DELIVERED"]);
    expect(frames[0].step).toBeNull();
    expect(frames.map((f) => f.done)).toEqual([false, false, false, false, true]);
  });

  it("stays in the state where a rule was broken", () => {
    const frames = playbackFrames(walkPath(["CONFIRM_PAYMENT", "SHIP_ITEMS"], FSM), FSM.initial_state);
    expect(frames).toHaveLength(3);
    expect(frames[2]).toMatchObject({ state: "PAID", done: true });
    expect(frames[2].step?.ok).toBe(false);
  });

  it("an empty order is a single finished frame", () => {
    expect(playbackFrames([], FSM.initial_state)).toEqual([{ index: 0, state: "ORDER_CREATED", step: null, done: true }]);
  });
});

describe("analysisStages: the checks in the order the backend runs them", () => {
  const statuses = (name: keyof typeof RECORDS) => analysisStages(RECORDS[name]).map((s) => [s.key, s.status]);

  it("a rule violation fails at the rules and skips both models", () => {
    expect(statuses("rule_violation")).toEqual([
      ["input", "pass"], ["rules", "fail"], ["finished", "skipped"], ["path", "skipped"], ["timing", "skipped"],
      ["verdict", "fail"],
    ]);
    expect(analysisStages(RECORDS.rule_violation)[1].detail).toContain("SHIP_ITEMS");
  });

  it("an unfinished order waits after the rules", () => {
    expect(statuses("incomplete")).toEqual([
      ["input", "pass"], ["rules", "pass"], ["finished", "waiting"], ["path", "skipped"], ["timing", "skipped"],
      ["verdict", "waiting"],
    ]);
  });

  it("a normal order clears both models", () => {
    expect(statuses("normal_timed")).toEqual([
      ["input", "pass"], ["rules", "pass"], ["finished", "pass"], ["path", "clear"], ["timing", "clear"],
      ["verdict", "clear"],
    ]);
  });

  it("each model's flag shows on its own stage", () => {
    expect(statuses("path_anomaly").slice(3)).toEqual([["path", "flagged"], ["timing", "skipped"], ["verdict", "flagged"]]);
    expect(statuses("timing_anomaly").slice(3)).toEqual([["path", "clear"], ["timing", "flagged"], ["verdict", "flagged"]]);
  });

  it("details quote the backend's own numbers and reasons", () => {
    const timing = analysisStages(RECORDS.timing_anomaly)[4];
    expect(timing.detail).toContain(RECORDS.timing_anomaly.timing_evaluation!.anomaly_score.toFixed(2));
    // The fixture order is flagged by the cluster check only; a normal order by none.
    expect(RECORDS.timing_anomaly.timing_evaluation!.checks.filter((c) => c.is_flagged).map((c) => c.name)).toEqual(["cluster"]);
    expect(timing.detail).toContain("flagged by: cluster");
    expect(analysisStages(RECORDS.normal_timed)[4].detail).toContain("no check flagged");
    const skipped = analysisStages(RECORDS.path_anomaly)[4];
    expect(skipped.detail).toBe(RECORDS.path_anomaly.timing_skip_reason);
  });
});
