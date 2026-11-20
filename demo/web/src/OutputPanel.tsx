import type { BatchEntry, DisplayGraph, Evaluation, Report } from "./model";

export function OutputPanel({
  evaluation,
  batch,
  graph,
  selectedSourceId,
  onSelectSource,
}: {
  evaluation: Evaluation | null;
  batch: BatchEntry[] | null;
  graph: DisplayGraph;
  selectedSourceId: string | null;
  onSelectSource: (sourceId: string) => void;
}) {
  if (evaluation === null && batch === null) {
    return null;
  }
  return (
    <section className="result" aria-label="Result">
      <h2>Result</h2>
      {batch !== null ? (
        <ul className="batch">
          {batch.map((entry) => (
            <li key={entry.id}>
              <span>{entry.id}</span>
              <span>{entry.title}</span>
              <span>{outcomeLabel(entry.output)}</span>
            </li>
          ))}
        </ul>
      ) : null}
      {evaluation === null ? null : evaluation.kind === "error" ? (
        <ErrorView outcome={evaluation.error.outcome} code={evaluation.error.code} message={evaluation.error.message} />
      ) : (
        <ReportView
          report={evaluation.report}
          graph={graph}
          selectedSourceId={selectedSourceId}
          onSelectSource={onSelectSource}
        />
      )}
    </section>
  );
}

function ErrorView({
  outcome,
  code,
  message,
}: {
  outcome: string;
  code: string;
  message: string;
}) {
  return (
    <div className="error-result">
      <p>
        {outcome} / {code}
      </p>
      <p>{message}</p>
    </div>
  );
}

function ReportView({
  report,
  graph,
  selectedSourceId,
  onSelectSource,
}: {
  report: Report;
  graph: DisplayGraph;
  selectedSourceId: string | null;
  onSelectSource: (sourceId: string) => void;
}) {
  const score =
    report.overall.adherence_score === null ? "none" : String(report.overall.adherence_score);
  return (
    <div className="report">
      <dl className="summary">
        <div>
          <dt>Verdict</dt>
          <dd>{report.overall.verdict}</dd>
        </div>
        <div>
          <dt>Score</dt>
          <dd>{score}</dd>
        </div>
        <div>
          <dt>Path</dt>
          <dd>{report.overall.path_validity}</dd>
        </div>
        <div>
          <dt>Expected</dt>
          <dd>{pathText(report.expected_path)}</dd>
        </div>
        <div>
          <dt>Observed</dt>
          <dd>{pathText(report.observed_path)}</dd>
        </div>
      </dl>
      <h3>Nodes</h3>
      <ul className="node-results">
        {Object.entries(report.node_results).map(([nodeId, result]) => (
          <li key={nodeId}>
            <button
              type="button"
              className={selectedSourceId === nodeId ? "is-selected" : ""}
              onClick={() => {
                onSelectSource(nodeId);
              }}
            >
              <span className="node-result-name">
                {nodeId} {labelFor(graph, nodeId)}
              </span>
              <span className={`status-pill status-${result.status}`}>
                {result.status}
                {result.score === null ? "" : ` ${result.score}`}
              </span>
            </button>
            <p className="node-result-note">{result.explanation}</p>
            {selectedSourceId === nodeId ? (
              <ul className="criteria">
                {result.criteria.map((criterion) => (
                  <li key={criterion.id}>
                    <span>
                      {criterion.id}: {criterion.result}
                    </span>
                    <span>Turns {criterion.evidence_turns.join(", ") || "none"}</span>
                    <span>{criterion.reason}</span>
                  </li>
                ))}
              </ul>
            ) : null}
          </li>
        ))}
      </ul>
      <h3>Transitions</h3>
      <ul>
        {report.transitions.map((hop) => (
          <li key={`${hop.source}-${hop.target}`}>
            {hop.source} to {hop.target}: {hop.assessment}
          </li>
        ))}
      </ul>
      <h3>Deviations</h3>
      <NoteList notes={report.deviations} empty="No deviations." />
      <h3>Uncertainty</h3>
      <NoteList notes={report.uncertainty} empty="No uncertainty." />
    </div>
  );
}

function NoteList({
  notes,
  empty,
}: {
  notes: Report["deviations"];
  empty: string;
}) {
  if (notes.length === 0) {
    return <p>{empty}</p>;
  }
  return (
    <ul>
      {notes.map((note, index) => (
        <li key={`${note.kind}-${note.node_id ?? ""}-${note.criterion ?? ""}-${index}`}>
          <span>{note.kind}</span>
          <span>{note.reason}</span>
        </li>
      ))}
    </ul>
  );
}

function labelFor(graph: DisplayGraph, sourceId: string): string {
  for (const node of graph.nodes) {
    if (node.sourceId === sourceId) {
      return node.label;
    }
  }
  return "";
}

function pathText(path: string[] | null): string {
  if (path === null || path.length === 0) {
    return "none";
  }
  return path.join(" -> ");
}

export function outcomeLabel(evaluation: Evaluation): string {
  if (evaluation.kind === "error") {
    return evaluation.error.code;
  }
  const score = evaluation.report.overall.adherence_score;
  const scoreText = score === null ? "none" : String(score);
  return `${evaluation.report.overall.verdict}, ${scoreText}, ${evaluation.report.overall.path_validity}`;
}
