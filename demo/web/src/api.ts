import { decodeJson, type JsonValue, type Read } from "./json";
import type { BatchEntry, CaseInputs, CaseSummary, Evaluation } from "./model";
import { parseBatch, parseCaseInputs, parseCases, parseEvaluation } from "./parse";

async function requestJson(path: string, init?: RequestInit): Promise<Read<JsonValue>> {
  try {
    const response = await fetch(path, init);
    const text = await response.text();
    return decodeJson(text, "The response");
  } catch {
    return { ok: false, message: "The demo service could not be reached." };
  }
}

export async function loadCases(): Promise<Read<CaseSummary[]>> {
  const decoded = await requestJson("/api/cases");
  if (!decoded.ok) {
    return decoded;
  }
  return parseCases(decoded.value);
}

export async function loadCase(caseId: string): Promise<Read<CaseInputs>> {
  const decoded = await requestJson(`/api/cases/${encodeURIComponent(caseId)}`);
  if (!decoded.ok) {
    return decoded;
  }
  return parseCaseInputs(decoded.value);
}

export async function runTexts(graph: string, transcript: string): Promise<Read<Evaluation>> {
  const body: JsonValue = { graph, transcript };
  const decoded = await requestJson("/api/run", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!decoded.ok) {
    return decoded;
  }
  return parseEvaluation(decoded.value);
}

export async function runAllCases(): Promise<Read<BatchEntry[]>> {
  const decoded = await requestJson("/api/runs", { method: "POST" });
  if (!decoded.ok) {
    return decoded;
  }
  return parseBatch(decoded.value);
}
