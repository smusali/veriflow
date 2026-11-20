import {
  decodeJson,
  fail,
  isRecord,
  readArray,
  readBoolean,
  readField,
  readIntegers,
  readMember,
  readNullableNumber,
  readNullableString,
  readNumber,
  readObject,
  readString,
  readStrings,
  type JsonValue,
  type Read,
} from "./json";
import {
  APPLICABILITY,
  ASSESSMENTS,
  FINDING_RESULTS,
  NODE_STATUSES,
  NODE_TYPES,
  PATH_VALIDITY,
  VERDICTS,
  type BatchEntry,
  type CaseInputs,
  type CaseSummary,
  type Criterion,
  type DisplayEdge,
  type DisplayGraph,
  type DisplayNode,
  type DisplayTranscript,
  type DisplayTurn,
  type Evaluation,
  type NodeResult,
  type NodeType,
  type Note,
  type Point,
  type Report,
  type Transition,
} from "./model";

function point(value: JsonValue, location: string): Read<Point> {
  const fields = readObject(value, location);
  if (!fields.ok) {
    return fields;
  }
  const x = readField(fields.value, "x", location);
  if (!x.ok) {
    return x;
  }
  const y = readField(fields.value, "y", location);
  if (!y.ok) {
    return y;
  }
  const xNumber = readNumber(x.value, `${location}.x`);
  if (!xNumber.ok) {
    return xNumber;
  }
  const yNumber = readNumber(y.value, `${location}.y`);
  if (!yNumber.ok) {
    return yNumber;
  }
  return { ok: true, value: { x: xNumber.value, y: yNumber.value } };
}

function nodeType(value: string): NodeType | null {
  for (const item of NODE_TYPES) {
    if (item === value) {
      return item;
    }
  }
  return null;
}

export function parseGraph(text: string): DisplayGraph {
  if (text.trim() === "") {
    return { nodes: [], edges: [], notices: [] };
  }
  const decoded = readObjectDocument(text, "The graph");
  if (!decoded.ok) {
    return { nodes: [], edges: [], notices: [decoded.message] };
  }
  const fields = decoded.value;
  const nodesField = readField(fields, "nodes", "graph");
  if (!nodesField.ok) {
    return { nodes: [], edges: [], notices: [nodesField.message] };
  }
  const edgesField = readField(fields, "edges", "graph");
  if (!edgesField.ok) {
    return { nodes: [], edges: [], notices: [edgesField.message] };
  }
  const nodeItems = readArray(nodesField.value, "graph.nodes");
  if (!nodeItems.ok) {
    return { nodes: [], edges: [], notices: [nodeItems.message] };
  }
  const edgeItems = readArray(edgesField.value, "graph.edges");
  if (!edgeItems.ok) {
    return { nodes: [], edges: [], notices: [edgeItems.message] };
  }

  const notices: string[] = [];
  const nodes: DisplayNode[] = [];
  const counts = new Map<string, number>();
  for (let index = 0; index < nodeItems.value.length; index += 1) {
    const item = nodeItems.value[index];
    if (item === undefined) {
      notices.push(`nodes[${index}] is missing.`);
      continue;
    }
    const parsed = parseNode(item, index, counts);
    notices.push(...parsed.notices);
    if (parsed.node !== null) {
      nodes.push(parsed.node);
    }
  }

  const firstFlow = new Map<string, string>();
  for (const node of nodes) {
    if (!firstFlow.has(node.sourceId)) {
      firstFlow.set(node.sourceId, node.flowId);
    }
  }

  const edges: DisplayEdge[] = [];
  const edgeIds = new Set<string>();
  for (let index = 0; index < edgeItems.value.length; index += 1) {
    const item = edgeItems.value[index];
    if (item === undefined) {
      notices.push(`edges[${index}] is missing.`);
      continue;
    }
    const parsed = parseEdge(item, index, firstFlow, edgeIds);
    notices.push(...parsed.notices);
    if (parsed.edge !== null) {
      edges.push(parsed.edge);
    }
  }
  return { nodes, edges, notices };
}

function parseNode(
  value: JsonValue,
  index: number,
  counts: Map<string, number>,
): { node: DisplayNode | null; notices: string[] } {
  const location = `nodes[${index}]`;
  const fields = readObject(value, location);
  if (!fields.ok) {
    return { node: null, notices: [fields.message] };
  }
  const id = readField(fields.value, "id", location);
  if (!id.ok) {
    return { node: null, notices: [id.message] };
  }
  const idText = readString(id.value, `${location}.id`);
  if (!idText.ok) {
    return { node: null, notices: [idText.message] };
  }
  if (idText.value === "") {
    return { node: null, notices: [`${location}.id must be a non-empty string.`] };
  }
  const typeField = readField(fields.value, "type", location);
  if (!typeField.ok) {
    return { node: null, notices: [typeField.message] };
  }
  const typeText = readString(typeField.value, `${location}.type`);
  if (!typeText.ok) {
    return { node: null, notices: [typeText.message] };
  }
  const notices: string[] = [];
  const kind = nodeType(typeText.value);
  if (kind === null) {
    notices.push(`${location}.type ${typeText.value} is unsupported.`);
  }
  const dataField = readField(fields.value, "data", location);
  if (!dataField.ok) {
    return { node: null, notices: [...notices, dataField.message] };
  }
  const data = readObject(dataField.value, `${location}.data`);
  if (!data.ok) {
    return { node: null, notices: [...notices, data.message] };
  }
  const labelField = readField(data.value, "label", `${location}.data`);
  if (!labelField.ok) {
    return { node: null, notices: [...notices, labelField.message] };
  }
  const label = readString(labelField.value, `${location}.data.label`);
  if (!label.ok) {
    return { node: null, notices: [...notices, label.message] };
  }
  const descriptionField = readField(data.value, "description", `${location}.data`);
  if (!descriptionField.ok) {
    return { node: null, notices: [...notices, descriptionField.message] };
  }
  const description = readString(descriptionField.value, `${location}.data.description`);
  if (!description.ok) {
    return { node: null, notices: [...notices, description.message] };
  }
  let position: Point | null = null;
  if (Object.hasOwn(fields.value, "position")) {
    const raw = fields.value.position;
    if (raw !== undefined) {
      const parsed = point(raw, `${location}.position`);
      if (parsed.ok) {
        position = parsed.value;
      } else {
        notices.push(parsed.message);
      }
    }
  }
  const seen = counts.get(idText.value) ?? 0;
  counts.set(idText.value, seen + 1);
  if (seen > 0) {
    notices.push(`${location}.id ${idText.value} is duplicated. Edges use the first copy.`);
  }
  return {
    node: {
      flowId: `n${index}`,
      sourceId: idText.value,
      nodeType: kind ?? "process",
      label: label.value,
      description: description.value,
      position,
    },
    notices,
  };
}

function parseEdge(
  value: JsonValue,
  index: number,
  firstFlow: Map<string, string>,
  edgeIds: Set<string>,
): { edge: DisplayEdge | null; notices: string[] } {
  const location = `edges[${index}]`;
  const fields = readObject(value, location);
  if (!fields.ok) {
    return { edge: null, notices: [fields.message] };
  }
  const id = readField(fields.value, "id", location);
  if (!id.ok) {
    return { edge: null, notices: [id.message] };
  }
  const idText = readString(id.value, `${location}.id`);
  if (!idText.ok) {
    return { edge: null, notices: [idText.message] };
  }
  if (idText.value === "") {
    return { edge: null, notices: [`${location}.id must be a non-empty string.`] };
  }
  const source = readField(fields.value, "source", location);
  if (!source.ok) {
    return { edge: null, notices: [source.message] };
  }
  const sourceText = readString(source.value, `${location}.source`);
  if (!sourceText.ok) {
    return { edge: null, notices: [sourceText.message] };
  }
  const target = readField(fields.value, "target", location);
  if (!target.ok) {
    return { edge: null, notices: [target.message] };
  }
  const targetText = readString(target.value, `${location}.target`);
  if (!targetText.ok) {
    return { edge: null, notices: [targetText.message] };
  }
  const sourceFlow = firstFlow.get(sourceText.value);
  const targetFlow = firstFlow.get(targetText.value);
  const notices: string[] = [];
  if (sourceFlow === undefined || targetFlow === undefined) {
    notices.push(`${location} does not match a node id.`);
    return { edge: null, notices };
  }
  let edgeId = idText.value;
  if (edgeIds.has(edgeId)) {
    edgeId = `${idText.value}#${index}`;
    notices.push(`${location}.id ${idText.value} is duplicated.`);
  }
  edgeIds.add(edgeId);
  return {
    edge: {
      id: edgeId,
      sourceId: sourceText.value,
      targetId: targetText.value,
      source: sourceFlow,
      target: targetFlow,
    },
    notices,
  };
}

export function parseTranscript(text: string): DisplayTranscript {
  if (text.trim() === "") {
    return { turns: [], notices: [] };
  }
  const decoded = readArrayDocument(text, "The transcript");
  if (!decoded.ok) {
    return { turns: [], notices: [decoded.message] };
  }
  const notices: string[] = [];
  const turns: DisplayTurn[] = [];
  for (let index = 0; index < decoded.value.length; index += 1) {
    const item = decoded.value[index];
    if (item === undefined) {
      notices.push(`turns[${index}] is missing.`);
      continue;
    }
    const parsed = parseTurn(item, index);
    if (!parsed.ok) {
      notices.push(parsed.message);
      continue;
    }
    turns.push(parsed.value);
  }
  return { turns: markOverlaps(turns), notices };
}

function parseTurn(value: JsonValue, index: number): Read<DisplayTurn> {
  const location = `turns[${index}]`;
  const fields = readObject(value, location);
  if (!fields.ok) {
    return fields;
  }
  const role = readField(fields.value, "role", location);
  if (!role.ok) {
    return role;
  }
  const roleText = readString(role.value, `${location}.role`);
  if (!roleText.ok) {
    return roleText;
  }
  if (roleText.value !== "user" && roleText.value !== "assistant") {
    return fail(`${location}.role ${roleText.value} is unsupported.`);
  }
  const content = readField(fields.value, "content", location);
  if (!content.ok) {
    return content;
  }
  const contentText = readString(content.value, `${location}.content`);
  if (!contentText.ok) {
    return contentText;
  }
  const beginning = readField(fields.value, "beginning", location);
  if (!beginning.ok) {
    return beginning;
  }
  const start = readNumber(beginning.value, `${location}.beginning`);
  if (!start.ok) {
    return start;
  }
  const endField = readField(fields.value, "end", location);
  if (!endField.ok) {
    return endField;
  }
  const end = readNumber(endField.value, `${location}.end`);
  if (!end.ok) {
    return end;
  }
  if (start.value < 0 || end.value < 0) {
    return fail(`${location} has a negative timestamp.`);
  }
  if (end.value < start.value) {
    return fail(`${location}.end is earlier than its beginning.`);
  }
  return {
    ok: true,
    value: {
      index,
      role: roleText.value,
      content: contentText.value,
      beginning: start.value,
      end: end.value,
      overlaps: [],
    },
  };
}

function markOverlaps(turns: DisplayTurn[]): DisplayTurn[] {
  return turns.map((turn) => {
    const overlaps: number[] = [];
    for (const other of turns) {
      if (other.index === turn.index) {
        continue;
      }
      if (turn.beginning < other.end && other.beginning < turn.end) {
        overlaps.push(other.index);
      }
    }
    return { ...turn, overlaps };
  });
}

function readObjectDocument(
  text: string,
  label: string,
): Read<{ [key: string]: JsonValue }> {
  const decoded = decodeDocument(text, label);
  if (!decoded.ok) {
    return decoded;
  }
  if (!isRecord(decoded.value)) {
    return fail(`${label} must be a JSON object.`);
  }
  return { ok: true, value: decoded.value };
}

function readArrayDocument(text: string, label: string): Read<JsonValue[]> {
  const decoded = decodeDocument(text, label);
  if (!decoded.ok) {
    return decoded;
  }
  return readArray(decoded.value, label);
}

function decodeDocument(text: string, label: string): Read<JsonValue> {
  return decodeJson(text, label);
}

export function parseCases(value: JsonValue): Read<CaseSummary[]> {
  const fields = readObject(value, "cases");
  if (!fields.ok) {
    return fields;
  }
  const items = readField(fields.value, "cases", "cases");
  if (!items.ok) {
    return items;
  }
  const list = readArray(items.value, "cases.cases");
  if (!list.ok) {
    return list;
  }
  const cases: CaseSummary[] = [];
  for (let index = 0; index < list.value.length; index += 1) {
    const item = list.value[index];
    if (item === undefined) {
      return fail(`cases.cases[${index}] is missing.`);
    }
    const parsed = parseCaseSummary(item, index);
    if (!parsed.ok) {
      return parsed;
    }
    cases.push(parsed.value);
  }
  return { ok: true, value: cases };
}

function parseCaseSummary(value: JsonValue, index: number): Read<CaseSummary> {
  const location = `cases.cases[${index}]`;
  const fields = readObject(value, location);
  if (!fields.ok) {
    return fields;
  }
  const id = requireString(fields.value, "id", location);
  if (!id.ok) {
    return id;
  }
  const title = requireString(fields.value, "title", location);
  if (!title.ok) {
    return title;
  }
  const category = requireString(fields.value, "category", location);
  if (!category.ok) {
    return category;
  }
  const graphPresent = requireBoolean(fields.value, "graph_present", location);
  if (!graphPresent.ok) {
    return graphPresent;
  }
  const transcriptPresent = requireBoolean(fields.value, "transcript_present", location);
  if (!transcriptPresent.ok) {
    return transcriptPresent;
  }
  const runnable = requireBoolean(fields.value, "runnable", location);
  if (!runnable.ok) {
    return runnable;
  }
  const notes = requireNullableString(fields.value, "notes", location);
  if (!notes.ok) {
    return notes;
  }
  return {
    ok: true,
    value: {
      id: id.value,
      title: title.value,
      category: category.value,
      graph_present: graphPresent.value,
      transcript_present: transcriptPresent.value,
      runnable: runnable.value,
      notes: notes.value,
    },
  };
}

export function parseCaseInputs(value: JsonValue): Read<CaseInputs> {
  const fields = readObject(value, "case");
  if (!fields.ok) {
    return fields;
  }
  const id = requireString(fields.value, "id", "case");
  if (!id.ok) {
    return id;
  }
  const title = requireString(fields.value, "title", "case");
  if (!title.ok) {
    return title;
  }
  const category = requireString(fields.value, "category", "case");
  if (!category.ok) {
    return category;
  }
  const notes = requireNullableString(fields.value, "notes", "case");
  if (!notes.ok) {
    return notes;
  }
  const graphText = requireNullableString(fields.value, "graph_text", "case");
  if (!graphText.ok) {
    return graphText;
  }
  const transcriptText = requireNullableString(fields.value, "transcript_text", "case");
  if (!transcriptText.ok) {
    return transcriptText;
  }
  return {
    ok: true,
    value: {
      id: id.value,
      title: title.value,
      category: category.value,
      notes: notes.value,
      graph_text: graphText.value,
      transcript_text: transcriptText.value,
    },
  };
}

export function parseEvaluation(value: JsonValue): Read<Evaluation> {
  const fields = readObject(value, "result");
  if (!fields.ok) {
    return fields;
  }
  if (Object.hasOwn(fields.value, "error")) {
    const error = fields.value.error;
    if (error === undefined) {
      return fail("result.error is required.");
    }
    return parseError(error);
  }
  const report = parseReport(fields.value);
  if (!report.ok) {
    return report;
  }
  return { ok: true, value: { kind: "report", report: report.value } };
}

function parseError(value: JsonValue): Read<Evaluation> {
  const fields = readObject(value, "result.error");
  if (!fields.ok) {
    return fields;
  }
  const outcome = requireString(fields.value, "outcome", "result.error");
  if (!outcome.ok) {
    return outcome;
  }
  const code = requireString(fields.value, "code", "result.error");
  if (!code.ok) {
    return code;
  }
  const message = requireString(fields.value, "message", "result.error");
  if (!message.ok) {
    return message;
  }
  return {
    ok: true,
    value: {
      kind: "error",
      error: { outcome: outcome.value, code: code.value, message: message.value },
    },
  };
}

function parseReport(fields: { [key: string]: JsonValue }): Read<Report> {
  const overall = readField(fields, "overall", "result");
  if (!overall.ok) {
    return overall;
  }
  const overallFields = readObject(overall.value, "result.overall");
  if (!overallFields.ok) {
    return overallFields;
  }
  const score = requireNullableNumber(overallFields.value, "adherence_score", "result.overall");
  if (!score.ok) {
    return score;
  }
  const verdict = requireMember(overallFields.value, "verdict", "result.overall", VERDICTS);
  if (!verdict.ok) {
    return verdict;
  }
  const validity = requireMember(
    overallFields.value,
    "path_validity",
    "result.overall",
    PATH_VALIDITY,
  );
  if (!validity.ok) {
    return validity;
  }
  const expected = readField(fields, "expected_path", "result");
  if (!expected.ok) {
    return expected;
  }
  let expectedPath: Read<string[] | null>;
  if (expected.value === null) {
    expectedPath = { ok: true, value: null };
  } else {
    expectedPath = readStrings(expected.value, "result.expected_path");
  }
  if (!expectedPath.ok) {
    return expectedPath;
  }
  const observed = requireStrings(fields, "observed_path", "result");
  if (!observed.ok) {
    return observed;
  }
  const transitions = parseTransitions(fields);
  if (!transitions.ok) {
    return transitions;
  }
  const nodeResults = parseNodeResults(fields);
  if (!nodeResults.ok) {
    return nodeResults;
  }
  const deviations = parseNotes(fields, "deviations");
  if (!deviations.ok) {
    return deviations;
  }
  const uncertainty = parseNotes(fields, "uncertainty");
  if (!uncertainty.ok) {
    return uncertainty;
  }
  return {
    ok: true,
    value: {
      overall: {
        adherence_score: score.value,
        verdict: verdict.value,
        path_validity: validity.value,
      },
      expected_path: expectedPath.value,
      observed_path: observed.value,
      transitions: transitions.value,
      node_results: nodeResults.value,
      deviations: deviations.value,
      uncertainty: uncertainty.value,
    },
  };
}

function parseTransitions(fields: { [key: string]: JsonValue }): Read<Transition[]> {
  const raw = readField(fields, "transitions", "result");
  if (!raw.ok) {
    return raw;
  }
  const items = readArray(raw.value, "result.transitions");
  if (!items.ok) {
    return items;
  }
  const transitions: Transition[] = [];
  for (let index = 0; index < items.value.length; index += 1) {
    const item = items.value[index];
    if (item === undefined) {
      return fail(`result.transitions[${index}] is missing.`);
    }
    const location = `result.transitions[${index}]`;
    const record = readObject(item, location);
    if (!record.ok) {
      return record;
    }
    const source = requireString(record.value, "source", location);
    if (!source.ok) {
      return source;
    }
    const target = requireString(record.value, "target", location);
    if (!target.ok) {
      return target;
    }
    const assessment = requireMember(record.value, "assessment", location, ASSESSMENTS);
    if (!assessment.ok) {
      return assessment;
    }
    transitions.push({
      source: source.value,
      target: target.value,
      assessment: assessment.value,
    });
  }
  return { ok: true, value: transitions };
}

function parseNodeResults(fields: {
  [key: string]: JsonValue;
}): Read<{ [nodeId: string]: NodeResult }> {
  const raw = readField(fields, "node_results", "result");
  if (!raw.ok) {
    return raw;
  }
  const record = readObject(raw.value, "result.node_results");
  if (!record.ok) {
    return record;
  }
  const results: { [nodeId: string]: NodeResult } = {};
  for (const nodeId of Object.keys(record.value)) {
    const item = record.value[nodeId];
    if (item === undefined) {
      return fail(`result.node_results.${nodeId} is missing.`);
    }
    const parsed = parseNodeResult(item, nodeId);
    if (!parsed.ok) {
      return parsed;
    }
    results[nodeId] = parsed.value;
  }
  return { ok: true, value: results };
}

function parseNodeResult(value: JsonValue, nodeId: string): Read<NodeResult> {
  const location = `result.node_results.${nodeId}`;
  const fields = readObject(value, location);
  if (!fields.ok) {
    return fields;
  }
  const status = requireMember(fields.value, "status", location, NODE_STATUSES);
  if (!status.ok) {
    return status;
  }
  const applicability = requireMember(fields.value, "applicability", location, APPLICABILITY);
  if (!applicability.ok) {
    return applicability;
  }
  const score = requireNullableNumber(fields.value, "score", location);
  if (!score.ok) {
    return score;
  }
  const evidence = requireIntegers(fields.value, "evidence_turns", location);
  if (!evidence.ok) {
    return evidence;
  }
  const missing = requireStrings(fields.value, "missing_requirements", location);
  if (!missing.ok) {
    return missing;
  }
  const violated = requireStrings(fields.value, "violated_requirements", location);
  if (!violated.ok) {
    return violated;
  }
  const explanation = requireString(fields.value, "explanation", location);
  if (!explanation.ok) {
    return explanation;
  }
  const criteria = parseCriteria(fields.value, location);
  if (!criteria.ok) {
    return criteria;
  }
  return {
    ok: true,
    value: {
      status: status.value,
      applicability: applicability.value,
      score: score.value,
      evidence_turns: evidence.value,
      missing_requirements: missing.value,
      violated_requirements: violated.value,
      explanation: explanation.value,
      criteria: criteria.value,
    },
  };
}

function parseCriteria(
  fields: { [key: string]: JsonValue },
  location: string,
): Read<Criterion[]> {
  const raw = readField(fields, "criteria", location);
  if (!raw.ok) {
    return raw;
  }
  const items = readArray(raw.value, `${location}.criteria`);
  if (!items.ok) {
    return items;
  }
  const criteria: Criterion[] = [];
  for (let index = 0; index < items.value.length; index += 1) {
    const item = items.value[index];
    if (item === undefined) {
      return fail(`${location}.criteria[${index}] is missing.`);
    }
    const criterionLocation = `${location}.criteria[${index}]`;
    const record = readObject(item, criterionLocation);
    if (!record.ok) {
      return record;
    }
    const id = requireString(record.value, "id", criterionLocation);
    if (!id.ok) {
      return id;
    }
    const applicability = requireMember(
      record.value,
      "applicability",
      criterionLocation,
      APPLICABILITY,
    );
    if (!applicability.ok) {
      return applicability;
    }
    const result = requireMember(record.value, "result", criterionLocation, FINDING_RESULTS);
    if (!result.ok) {
      return result;
    }
    const evidence = requireIntegers(record.value, "evidence_turns", criterionLocation);
    if (!evidence.ok) {
      return evidence;
    }
    const missing = requireStrings(record.value, "missing_requirements", criterionLocation);
    if (!missing.ok) {
      return missing;
    }
    const violated = requireStrings(record.value, "violated_requirements", criterionLocation);
    if (!violated.ok) {
      return violated;
    }
    const reason = requireString(record.value, "reason", criterionLocation);
    if (!reason.ok) {
      return reason;
    }
    criteria.push({
      id: id.value,
      applicability: applicability.value,
      result: result.value,
      evidence_turns: evidence.value,
      missing_requirements: missing.value,
      violated_requirements: violated.value,
      reason: reason.value,
    });
  }
  return { ok: true, value: criteria };
}

function parseNotes(
  fields: { [key: string]: JsonValue },
  name: "deviations" | "uncertainty",
): Read<Note[]> {
  const raw = readField(fields, name, "result");
  if (!raw.ok) {
    return raw;
  }
  const items = readArray(raw.value, `result.${name}`);
  if (!items.ok) {
    return items;
  }
  const notes: Note[] = [];
  for (let index = 0; index < items.value.length; index += 1) {
    const item = items.value[index];
    if (item === undefined) {
      return fail(`result.${name}[${index}] is missing.`);
    }
    const location = `result.${name}[${index}]`;
    const record = readObject(item, location);
    if (!record.ok) {
      return record;
    }
    const kind = requireString(record.value, "kind", location);
    if (!kind.ok) {
      return kind;
    }
    const nodeId = requireNullableString(record.value, "node_id", location);
    if (!nodeId.ok) {
      return nodeId;
    }
    const criterion = requireNullableString(record.value, "criterion", location);
    if (!criterion.ok) {
      return criterion;
    }
    const source = requireNullableString(record.value, "source", location);
    if (!source.ok) {
      return source;
    }
    const target = requireNullableString(record.value, "target", location);
    if (!target.ok) {
      return target;
    }
    const reason = requireString(record.value, "reason", location);
    if (!reason.ok) {
      return reason;
    }
    notes.push({
      kind: kind.value,
      node_id: nodeId.value,
      criterion: criterion.value,
      source: source.value,
      target: target.value,
      reason: reason.value,
    });
  }
  return { ok: true, value: notes };
}

export function parseBatch(value: JsonValue): Read<BatchEntry[]> {
  const fields = readObject(value, "runs");
  if (!fields.ok) {
    return fields;
  }
  const raw = readField(fields.value, "runs", "runs");
  if (!raw.ok) {
    return raw;
  }
  const items = readArray(raw.value, "runs.runs");
  if (!items.ok) {
    return items;
  }
  const entries: BatchEntry[] = [];
  for (let index = 0; index < items.value.length; index += 1) {
    const item = items.value[index];
    if (item === undefined) {
      return fail(`runs.runs[${index}] is missing.`);
    }
    const location = `runs.runs[${index}]`;
    const record = readObject(item, location);
    if (!record.ok) {
      return record;
    }
    const id = requireString(record.value, "id", location);
    if (!id.ok) {
      return id;
    }
    const title = requireString(record.value, "title", location);
    if (!title.ok) {
      return title;
    }
    const output = readField(record.value, "output", location);
    if (!output.ok) {
      return output;
    }
    const evaluation = parseEvaluation(output.value);
    if (!evaluation.ok) {
      return evaluation;
    }
    entries.push({ id: id.value, title: title.value, output: evaluation.value });
  }
  return { ok: true, value: entries };
}

function requireString(
  fields: { [key: string]: JsonValue },
  name: string,
  location: string,
): Read<string> {
  const field = readField(fields, name, location);
  if (!field.ok) {
    return field;
  }
  return readString(field.value, `${location}.${name}`);
}

function requireNullableString(
  fields: { [key: string]: JsonValue },
  name: string,
  location: string,
): Read<string | null> {
  const field = readField(fields, name, location);
  if (!field.ok) {
    return field;
  }
  return readNullableString(field.value, `${location}.${name}`);
}

function requireNullableNumber(
  fields: { [key: string]: JsonValue },
  name: string,
  location: string,
): Read<number | null> {
  const field = readField(fields, name, location);
  if (!field.ok) {
    return field;
  }
  return readNullableNumber(field.value, `${location}.${name}`);
}

function requireBoolean(
  fields: { [key: string]: JsonValue },
  name: string,
  location: string,
): Read<boolean> {
  const field = readField(fields, name, location);
  if (!field.ok) {
    return field;
  }
  return readBoolean(field.value, `${location}.${name}`);
}

function requireStrings(
  fields: { [key: string]: JsonValue },
  name: string,
  location: string,
): Read<string[]> {
  const field = readField(fields, name, location);
  if (!field.ok) {
    return field;
  }
  return readStrings(field.value, `${location}.${name}`);
}

function requireIntegers(
  fields: { [key: string]: JsonValue },
  name: string,
  location: string,
): Read<number[]> {
  const field = readField(fields, name, location);
  if (!field.ok) {
    return field;
  }
  return readIntegers(field.value, `${location}.${name}`);
}

function requireMember<T extends string>(
  fields: { [key: string]: JsonValue },
  name: string,
  location: string,
  allowed: readonly T[],
): Read<T> {
  const field = readField(fields, name, location);
  if (!field.ok) {
    return field;
  }
  return readMember(field.value, `${location}.${name}`, allowed);
}
