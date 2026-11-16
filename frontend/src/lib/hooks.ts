import { createContext, useCallback, useContext, useEffect, useState } from "react";
import type { FsmDefinition } from "../types";

export interface AsyncState<T> {
  data: T | null;
  error: Error | null; // the full error (an ApiError keeps its checkpoint and detail lines)
  loading: boolean;
  reload: () => void;
}

/** Runs `load` on mount and whenever `deps` change; `reload()` runs it again. */
export function useAsync<T>(load: () => Promise<T>, deps: unknown[] = []): AsyncState<T> {
  const [data, setData] = useState<T | null>(null);
  const [error, setError] = useState<Error | null>(null);
  const [loading, setLoading] = useState(true);
  const [tick, setTick] = useState(0);

  useEffect(() => {
    let alive = true;
    setLoading(true);
    load()
      .then((d) => {
        if (alive) {
          setData(d);
          setError(null);
        }
      })
      .catch((e: unknown) => {
        if (alive) setError(e instanceof Error ? e : new Error(String(e)));
      })
      .finally(() => {
        if (alive) setLoading(false);
      });
    return () => {
      alive = false;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [...deps, tick]);

  const reload = useCallback(() => setTick((t) => t + 1), []);
  return { data, error, loading, reload };
}

/** The automaton definition, fetched once by App and shared with every page (with its error). */
export const FsmContext = createContext<AsyncState<FsmDefinition> | null>(null);

export function useFsmState(): AsyncState<FsmDefinition> {
  const state = useContext(FsmContext);
  if (state === null) throw new Error("[UI] expected an FsmContext provider above this component, got none; input=useFsmState");
  return state;
}
