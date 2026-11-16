import { useEffect, useMemo, useState } from "react";
import type { FsmDefinition, Measurement } from "../types";
import { describePercentile, formatDuration } from "../lib/format";
import { playbackFrames, type Frame } from "../lib/playback";
import { walkPath } from "../lib/workflow";
import { FsmDiagram, type PathHighlight } from "./FsmDiagram";
import { Icon } from "./Icon";

/** What the models said about one step (absent when the order was not scored, e.g. a shortest path). */
export interface StepFacts {
  probability: number;
  rareStep: boolean; // the Markov chain marked this kind of step unusual
  unusual: boolean;  // rare step or unusual duration
  duration: Measurement | null;
}

const SPEEDS = { "0.5×": 1800, "1×": 1000, "2×": 500 } as const; // milliseconds per step
type Speed = keyof typeof SPEEDS;
const TOKEN_SHARE = 0.7; // the moving dot uses this share of each step's time

interface Props {
  fsm: FsmDefinition;
  events: string[];
  facts: Map<number, StepFacts>;
  autoPlay: boolean;
}

function caption(frame: Frame, fsm: FsmDefinition, facts: Map<number, StepFacts>, last: boolean) {
  const s = frame.step;
  if (s === null) return <>Start: every order begins in <b className="mono">{fsm.initial_state}</b>.</>;
  if (!s.ok) {
    const allowed = fsm.transitions.filter((t) => t.from_state === s.from).map((t) => t.event);
    const why = !fsm.events.includes(s.event) ? "is not an event of this workflow"
      : fsm.terminal_states.includes(s.from) ? `came after the order had already ended in ${s.from}`
        : `is not allowed in ${s.from}. Allowed: ${allowed.join(", ")}`;
    return <span style={{ color: "var(--bad)" }}><b>Step {s.step}: ✕ {s.event}</b> {why}. The rules stop the order here.</span>;
  }
  const f = facts.get(s.step);
  return (
    <>
      <b>Step {s.step}: {s.event}</b> moves <span className="mono">{s.from}</span> → <span className="mono">{s.to}</span>.
      {f !== undefined && (
        <span className={f.rareStep ? "warn-text" : undefined}>
          {" "}Taken in {(f.probability * 100).toFixed(1)}% of normal visits to {s.from}{f.rareStep ? " (unusual)" : ""}.
        </span>
      )}
      {f !== undefined && f.duration !== null && (
        <span className={f.duration.is_unusual ? "warn-text" : undefined}>
          {" "}Took {formatDuration(f.duration.value)}, {describePercentile(f.duration.percentile, "seconds")}.
        </span>
      )}
      {last && (fsm.terminal_states.includes(frame.state)
        ? <> Finished in <b className="mono">{frame.state}</b>.</>
        : <> Still in progress in <b className="mono">{frame.state}</b>.</>)}
    </>
  );
}

export function PathPlayer({ fsm, events, facts, autoPlay }: Props) {
  const frames = useMemo(() => playbackFrames(walkPath(events, fsm), fsm.initial_state), [events, fsm]);
  const lastIndex = frames.length - 1;
  const [index, setIndex] = useState(autoPlay ? 0 : lastIndex);
  const [playing, setPlaying] = useState(autoPlay);
  const [speed, setSpeed] = useState<Speed>("1×");

  useEffect(() => {
    setIndex(autoPlay ? 0 : lastIndex);
    setPlaying(autoPlay);
  }, [frames, autoPlay, lastIndex]);

  useEffect(() => {
    if (!playing) return;
    if (index >= lastIndex) {
      setPlaying(false);
      return;
    }
    const id = window.setTimeout(() => setIndex((i) => i + 1), SPEEDS[speed]);
    return () => window.clearTimeout(id);
  }, [playing, index, lastIndex, speed]);

  const frame = frames[index];
  const shown = frames.slice(1, index + 1).map((f) => f.step!);
  const highlight: PathHighlight = {
    steps: shown.map((s) => ({ ...s, unusual: facts.get(s.step)?.unusual === true })),
    failure: frame.step !== null && !frame.step.ok ? { state: frame.step.from, event: frame.step.event } : undefined,
    current: frame.state,
  };
  const token = frame.step !== null && frame.step.ok
    ? { edge: `${frame.step.from}|${frame.step.event}`, key: index, ms: SPEEDS[speed] * TOKEN_SHARE }
    : undefined;

  const togglePlay = () => {
    if (index >= lastIndex) setIndex(0);
    setPlaying(!playing || index >= lastIndex);
  };

  return (
    <div className="player">
      <div className="player-bar">
        <div className="player-controls" role="group" aria-label="Path playback">
          <button className="btn sm ghost" onClick={() => { setIndex(0); setPlaying(true); }} aria-label="Replay from the start" title="Replay">
            <Icon name="restart" size={15} />
          </button>
          <button className="btn sm ghost" disabled={index === 0} onClick={() => { setPlaying(false); setIndex(index - 1); }}
            aria-label="Previous step" title="Previous step">
            <Icon name="stepBack" size={15} />
          </button>
          <button className="btn sm primary" onClick={togglePlay} aria-label={playing ? "Pause" : "Play"}>
            <Icon name={playing ? "pause" : "play"} size={15} /> {playing ? "Pause" : index >= lastIndex ? "Replay" : "Play"}
          </button>
          <button className="btn sm ghost" disabled={index >= lastIndex} onClick={() => { setPlaying(false); setIndex(index + 1); }}
            aria-label="Next step" title="Next step">
            <Icon name="stepForward" size={15} />
          </button>
          <select className="select sm" value={speed} onChange={(e) => setSpeed(e.target.value as Speed)} aria-label="Playback speed">
            {(Object.keys(SPEEDS) as Speed[]).map((s) => <option key={s}>{s}</option>)}
          </select>
          <span className="faint player-count">step {index} of {lastIndex}</span>
        </div>
        <p className="player-caption" aria-live="polite">{caption(frame, fsm, facts, frame.done)}</p>
      </div>
      <FsmDiagram fsm={fsm} highlight={highlight} token={token} />
    </div>
  );
}
