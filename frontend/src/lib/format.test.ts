import { describe, expect, it } from "vitest";
import { describePercentile, describeProbability, formatDuration } from "./format";

describe("format (same wording as the backend)", () => {
  it("formatDuration picks a readable unit", () => {
    expect(formatDuration(45)).toBe("45 s");
    expect(formatDuration(750)).toBe("12.5 min");
    expect(formatDuration(8 * 3600)).toBe("8.0 h");
    expect(formatDuration(3.1 * 86400)).toBe("3.1 days");
  });

  it("describeProbability matches backend/ml/markov/model.py", () => {
    expect(describeProbability(Math.log10(0.791))).toBe("79.1% of orders");
    expect(describeProbability(Math.log10(1 / 27300))).toBe("about 1 in 27,300 orders");
  });

  it("describePercentile matches backend/pipeline/analyzer.py", () => {
    expect(describePercentile(99.9, "seconds")).toBe("slower than 99.9% of normal orders");
    expect(describePercentile(0.5, "currency")).toBe("lower than 99.5% of normal orders");
    expect(describePercentile(98.5, "count")).toBe("more than 98.5% of normal orders");
  });
});
