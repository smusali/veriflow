export const NODE_STATUSES = [
  "completed",
  "partial",
  "incorrect",
  "skipped",
  "not_applicable",
  "not_reached",
] as const;

export const NODE_TYPES = ["start", "decision", "process", "end"] as const;

export const ASSESSMENTS = ["permitted", "invalid", "unresolved"] as const;

export const VERDICTS = [
  "completed",
  "partial",
  "failed",
  "incomplete",
  "indeterminate",
] as const;

export const PATH_VALIDITY = ["valid", "invalid", "incomplete", "indeterminate"] as const;

export const APPLICABILITY = ["applicable", "not_applicable", "uncertain"] as const;

export const FINDING_RESULTS = ["met", "unmet", "contradicted", "insufficient"] as const;

export type NodeStatus = (typeof NODE_STATUSES)[number];
export type NodeType = (typeof NODE_TYPES)[number];
export type Assessment = (typeof ASSESSMENTS)[number];
export type Verdict = (typeof VERDICTS)[number];
export type PathValidity = (typeof PATH_VALIDITY)[number];
export type Applicability = (typeof APPLICABILITY)[number];
export type FindingResult = (typeof FINDING_RESULTS)[number];

export type Criterion = {
  id: string;
  applicability: Applicability;
  result: FindingResult;
  evidence_turns: number[];
  missing_requirements: string[];
  violated_requirements: string[];
  reason: string;
};

export type NodeResult = {
  status: NodeStatus;
  applicability: Applicability;
  score: number | null;
  evidence_turns: number[];
  missing_requirements: string[];
  violated_requirements: string[];
  explanation: string;
  criteria: Criterion[];
};

export type Transition = {
  source: string;
  target: string;
  assessment: Assessment;
};

export type Note = {
  kind: string;
  node_id: string | null;
  criterion: string | null;
  source: string | null;
  target: string | null;
  reason: string;
};

export type Report = {
  overall: {
    adherence_score: number | null;
    verdict: Verdict;
    path_validity: PathValidity;
  };
  expected_path: string[] | null;
  observed_path: string[];
  transitions: Transition[];
  node_results: { [nodeId: string]: NodeResult };
  deviations: Note[];
  uncertainty: Note[];
};

export type RunError = {
  outcome: string;
  code: string;
  message: string;
};

export type Evaluation = { kind: "report"; report: Report } | { kind: "error"; error: RunError };

export type CaseSummary = {
  id: string;
  title: string;
  category: string;
  graph_present: boolean;
  transcript_present: boolean;
  runnable: boolean;
  notes: string | null;
};

export type CaseInputs = {
  id: string;
  title: string;
  category: string;
  notes: string | null;
  graph_text: string | null;
  transcript_text: string | null;
};

export type BatchEntry = {
  id: string;
  title: string;
  output: Evaluation;
};

export type Point = { x: number; y: number };

export type DisplayNode = {
  flowId: string;
  sourceId: string;
  nodeType: NodeType;
  label: string;
  description: string;
  position: Point | null;
};

export type DisplayEdge = {
  id: string;
  sourceId: string;
  targetId: string;
  source: string;
  target: string;
};

export type DisplayGraph = {
  nodes: DisplayNode[];
  edges: DisplayEdge[];
  notices: string[];
};

export type DisplayTurn = {
  index: number;
  role: "user" | "assistant";
  content: string;
  beginning: number;
  end: number;
  overlaps: number[];
};

export type DisplayTranscript = {
  turns: DisplayTurn[];
  notices: string[];
};
