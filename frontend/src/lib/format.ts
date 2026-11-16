import type { MeasurementUnit } from "../types";

// Same duration rules as backend/ml/timing/features.py:format_duration.
export function formatDuration(seconds: number): string {
  if (seconds < 60) return `${Math.round(seconds)} s`;
  if (seconds < 3600) return `${(seconds / 60).toFixed(1)} min`;
  if (seconds < 2 * 86400) return `${(seconds / 3600).toFixed(1)} h`;
  return `${(seconds / 86400).toFixed(1)} days`;
}

export const formatNumber = (n: number, digits = 0) =>
  n.toLocaleString("en-US", { minimumFractionDigits: digits, maximumFractionDigits: digits });

export const formatAmount = (n: number) => n.toLocaleString("en-US", { maximumFractionDigits: 2 });

export const formatPercent = (fraction: number, digits = 1) => `${(fraction * 100).toFixed(digits)}%`;

// Same wording as backend/ml/markov/model.py:describe_probability.
export function describeProbability(log10p: number): string {
  const p = 10 ** log10p;
  if (p >= 0.01) return `${(p * 100).toFixed(1)}% of orders`;
  return `about 1 in ${formatNumber(Math.round(1 / p))} orders`;
}

const DIRECTION: Record<MeasurementUnit, [string, string]> = {
  seconds: ["slower", "faster"],
  currency: ["higher", "lower"],
  count: ["more", "fewer"],
};

// Same wording as backend/pipeline/analyzer.py:describe_measurement.
export function describePercentile(percentile: number, unit: MeasurementUnit): string {
  const [more, less] = DIRECTION[unit];
  return percentile >= 50
    ? `${more} than ${percentile.toFixed(1)}% of normal orders`
    : `${less} than ${(100 - percentile).toFixed(1)}% of normal orders`;
}

export function formatDateTime(iso: string): string {
  return new Date(iso).toLocaleString("en-GB", {
    day: "2-digit",
    month: "short",
    hour: "2-digit",
    minute: "2-digit",
    second: "2-digit",
  });
}

export function timeAgo(iso: string): string {
  const s = (Date.now() - new Date(iso).getTime()) / 1000;
  if (s < 45) return "just now";
  if (s < 3600) return `${Math.round(s / 60)} min ago`;
  if (s < 86400) return `${Math.round(s / 3600)} h ago`;
  return `${Math.round(s / 86400)} d ago`;
}

export function formatMeasurementValue(unit: MeasurementUnit, value: number): string {
  if (unit === "seconds") return formatDuration(value);
  if (unit === "currency") return formatAmount(value);
  return formatNumber(value);
}
