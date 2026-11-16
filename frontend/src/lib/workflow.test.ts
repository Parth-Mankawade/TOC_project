import { describe, expect, it } from "vitest";
import { FSM, HAPPY, RECORDS } from "../test/fixtures";
import { eventsOf, possibleOutcomes, walkPath } from "./workflow";

describe("walkPath replays events through the automaton like backend/fsm/validator.py", () => {
  it("follows a valid order to its final state", () => {
    const walk = walkPath(HAPPY, FSM);
    expect(walk.map((s) => [s.step, s.from, s.event, s.to, s.ok])).toEqual([
      [1, "ORDER_CREATED", "CONFIRM_PAYMENT", "PAID", true],
      [2, "PAID", "PACK_ITEMS", "PACKED", true],
      [3, "PACKED", "SHIP_ITEMS", "SHIPPED", true],
      [4, "SHIPPED", "DELIVER_ITEMS", "DELIVERED", true],
    ]);
  });

  it("stops at the first step that breaks a rule", () => {
    const walk = walkPath(["CONFIRM_PAYMENT", "SHIP_ITEMS", "DELIVER_ITEMS"], FSM);
    expect(walk).toHaveLength(2);
    expect(walk[1]).toMatchObject({ step: 2, from: "PAID", event: "SHIP_ITEMS", to: null, ok: false });
  });

  it("rejects an unknown event and any event after the order ended", () => {
    expect(walkPath(["HACK_CHECKOUT"], FSM)[0].ok).toBe(false);
    const after = walkPath([...HAPPY, "REFUND_ORDER"], FSM);
    expect(after[4]).toMatchObject({ from: "DELIVERED", ok: false });
  });

  it("agrees with the backend on where every recorded rule violation failed", () => {
    for (const record of [RECORDS.rule_violation, RECORDS.after_end]) {
      const walk = walkPath(eventsOf(record), FSM);
      const failed = walk[walk.length - 1];
      expect(failed.ok).toBe(false);
      expect(failed.step).toBe(record.fsm_evaluation.error?.failed_step);
    }
  });
});

describe("possibleOutcomes", () => {
  it("lists every ending still reachable, with the shortest way there", () => {
    expect(possibleOutcomes("PACKED", FSM)).toEqual({
      DELIVERED: ["SHIP_ITEMS", "DELIVER_ITEMS"],
      REFUNDED: ["REFUND_ORDER"],
    });
    expect(possibleOutcomes("DELIVERED", FSM)).toEqual({ DELIVERED: [] });
  });
});

describe("eventsOf", () => {
  it("reads both input formats exactly as stored", () => {
    expect(eventsOf(RECORDS.rule_violation)).toEqual(["CONFIRM_PAYMENT", "SHIP_ITEMS"]);
    expect(eventsOf(RECORDS.incomplete)).toEqual(["CONFIRM_PAYMENT", "PACK_ITEMS"]);
  });
});
