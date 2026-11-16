// Real backend output used by the frontend tests:
//   fsm.json      GET /api/v1/fsm (backend/api/routes.py: get_fsm_definition)
//   records.json  analysis records produced by the backend's AnomalyService with the trained models
import fsmJson from "./fsm.json";
import recordsJson from "./records.json";
import type { AnalysisRecord, FsmDefinition } from "../types";

export const FSM = fsmJson as FsmDefinition;

type RecordName = "rule_violation" | "after_end" | "incomplete" | "normal_timed" | "path_anomaly" | "timing_anomaly";
export const RECORDS = recordsJson as Record<RecordName, AnalysisRecord>;

export const HAPPY = ["CONFIRM_PAYMENT", "PACK_ITEMS", "SHIP_ITEMS", "DELIVER_ITEMS"];
