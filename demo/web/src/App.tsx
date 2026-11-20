import { useEffect, useMemo, useRef, useState } from "react";

import { loadCase, loadCases, runAllCases, runTexts } from "./api";
import type { LayoutMode } from "./layout";
import type { BatchEntry, CaseSummary, DisplayNode, Evaluation } from "./model";
import { OutputPanel } from "./OutputPanel";
import { parseGraph, parseTranscript } from "./parse";
import { TranscriptPane } from "./TranscriptPane";
import { WorkflowCanvas } from "./WorkflowCanvas";

type LoadedCase = {
  id: string;
  graphText: string;
  transcriptText: string;
};

export function App() {
  const [cases, setCases] = useState<CaseSummary[]>([]);
  const [listFault, setListFault] = useState<string | null>(null);
  const [caseFault, setCaseFault] = useState<string | null>(null);
  const [caseId, setCaseId] = useState("");
  const [notes, setNotes] = useState<string | null>(null);
  const [graphText, setGraphText] = useState("");
  const [transcriptText, setTranscriptText] = useState("");
  const [loaded, setLoaded] = useState<LoadedCase | null>(null);
  const [evaluation, setEvaluation] = useState<Evaluation | null>(null);
  const [fault, setFault] = useState<string | null>(null);
  const [batch, setBatch] = useState<BatchEntry[] | null>(null);
  const [running, setRunning] = useState(false);
  const [layoutMode, setLayoutMode] = useState<LayoutMode>("ranked");
  const [graphSourceOpen, setGraphSourceOpen] = useState(true);
  const [transcriptSourceOpen, setTranscriptSourceOpen] = useState(true);
  const [selectedFlowId, setSelectedFlowId] = useState<string | null>(null);
  const [selectedTurn, setSelectedTurn] = useState<number | null>(null);
  const loadToken = useRef(0);
  const runToken = useRef(0);

  useEffect(() => {
    let active = true;
    void loadCases().then((result) => {
      if (!active) {
        return;
      }
      if (!result.ok) {
        setListFault(result.message);
        return;
      }
      setCases(result.value);
    });
    return () => {
      active = false;
    };
  }, []);

  const graph = useMemo(() => parseGraph(graphText), [graphText]);
  const transcript = useMemo(() => parseTranscript(transcriptText), [transcriptText]);
  const selectedNode = graph.nodes.find((node) => node.flowId === selectedFlowId) ?? null;

  const highlightedNodes = useMemo(() => {
    const ids = new Set<string>();
    if (evaluation === null || evaluation.kind !== "report" || selectedTurn === null) {
      return ids;
    }
    for (const node of graph.nodes) {
      const result = evaluation.report.node_results[node.sourceId];
      if (result !== undefined && result.evidence_turns.includes(selectedTurn)) {
        ids.add(node.flowId);
      }
    }
    return ids;
  }, [evaluation, graph.nodes, selectedTurn]);

  const highlightedTurns = useMemo(() => {
    const indexes = new Set<number>();
    if (
      evaluation === null ||
      evaluation.kind !== "report" ||
      selectedNode === null
    ) {
      return indexes;
    }
    const result = evaluation.report.node_results[selectedNode.sourceId];
    if (result === undefined) {
      return indexes;
    }
    for (const index of result.evidence_turns) {
      indexes.add(index);
    }
    return indexes;
  }, [evaluation, selectedNode]);

  function clearResult() {
    runToken.current += 1;
    setRunning(false);
    setEvaluation(null);
    setFault(null);
    setBatch(null);
    setSelectedFlowId(null);
    setSelectedTurn(null);
  }

  async function onCase(nextId: string) {
    const token = loadToken.current + 1;
    loadToken.current = token;
    setCaseId(nextId);
    setCaseFault(null);
    clearResult();
    if (nextId === "") {
      setLoaded(null);
      setNotes(null);
      setGraphText("");
      setTranscriptText("");
      setGraphSourceOpen(true);
      setTranscriptSourceOpen(true);
      return;
    }
    const result = await loadCase(nextId);
    if (loadToken.current !== token) {
      return;
    }
    if (!result.ok) {
      setCaseFault(result.message);
      return;
    }
    const nextGraph = result.value.graph_text ?? "";
    const nextTranscript = result.value.transcript_text ?? "";
    setGraphText(nextGraph);
    setTranscriptText(nextTranscript);
    setGraphSourceOpen(nextGraph.trim() === "");
    setTranscriptSourceOpen(nextTranscript.trim() === "");
    setNotes(result.value.notes);
    setLoaded({ id: nextId, graphText: nextGraph, transcriptText: nextTranscript });
  }

  function onGraphText(value: string) {
    setGraphText(value);
    clearResult();
  }

  function onTranscriptText(value: string) {
    setTranscriptText(value);
    clearResult();
  }

  async function onRun() {
    const token = runToken.current + 1;
    runToken.current = token;
    setRunning(true);
    setFault(null);
    setEvaluation(null);
    setBatch(null);
    const result = await runTexts(graphText, transcriptText);
    if (runToken.current !== token) {
      return;
    }
    setRunning(false);
    if (!result.ok) {
      setFault(result.message);
      return;
    }
    setEvaluation(result.value);
  }

  async function onRunAll() {
    const token = runToken.current + 1;
    runToken.current = token;
    setRunning(true);
    setFault(null);
    setEvaluation(null);
    setBatch(null);
    const result = await runAllCases();
    if (runToken.current !== token) {
      return;
    }
    setRunning(false);
    if (!result.ok) {
      setFault(result.message);
      return;
    }
    setBatch(result.value);
    if (
      loaded !== null &&
      loaded.graphText === graphText &&
      loaded.transcriptText === transcriptText
    ) {
      const match = result.value.find((entry) => entry.id === loaded.id);
      if (match !== undefined) {
        setEvaluation(match.output);
      }
    }
  }

  const shown: Evaluation | null =
    fault === null
      ? evaluation
      : { kind: "error", error: { outcome: "evaluation_error", code: "unavailable", message: fault } };

  return (
    <div className="page">
      <header className="toolbar" aria-busy={running}>
        <div className="title-row">
          <h1>Veriflow</h1>
          <p>Choose a case, or paste a graph and a transcript. Run is what produces a result.</p>
        </div>
        <div className="controls">
          <label>
            Case
            <select
              value={caseId}
              onChange={(event) => {
                void onCase(event.target.value);
              }}
            >
              <option value="">Select a case</option>
              {cases.map((item) => (
                <option key={item.id} value={item.id}>
                  {item.id}: {item.title}
                </option>
              ))}
            </select>
          </label>
          <button className="primary" type="button" onClick={() => void onRun()} disabled={running}>
            {running ? "Running" : "Run"}
          </button>
          <button type="button" onClick={() => void onRunAll()} disabled={running}>
            Run all
          </button>
          <button
            type="button"
            onClick={() => {
              setLayoutMode(layoutMode === "ranked" ? "stored" : "ranked");
            }}
          >
            {layoutMode === "ranked" ? "Use file positions" : "Use ranked layout"}
          </button>
        </div>
        {listFault !== null ? <p className="notice">{listFault}</p> : null}
        {caseFault !== null ? <p className="notice">{caseFault}</p> : null}
        {notes !== null && notes !== "" ? <p className="notes">{notes}</p> : null}
      </header>
      <div className="workspace">
        <section className="pane canvas-column" aria-label="Workflow">
          <div className="pane-heading">
            <h2>Workflow</h2>
          </div>
          {graph.notices.map((notice) => (
            <p key={notice} className="notice">
              {notice}
            </p>
          ))}
          {graph.nodes.length === 0 ? null : (
            <div className="canvas-frame">
              <WorkflowCanvas
                graph={graph}
                evaluation={shown}
                layoutMode={layoutMode}
                highlighted={highlightedNodes}
                onSelect={(flowId) => {
                  setSelectedFlowId(flowId);
                  setSelectedTurn(null);
                }}
              />
            </div>
          )}
          {selectedNode !== null ? <NodeDetail node={selectedNode} /> : null}
          <details
            className="source"
            open={graphSourceOpen}
            onToggle={(event) => {
              setGraphSourceOpen(event.currentTarget.open);
            }}
          >
            <summary>Graph source</summary>
            <InputBlock label="Graph" text={graphText} onText={onGraphText} />
          </details>
        </section>
        <TranscriptPane
          transcript={transcript}
          highlighted={highlightedTurns}
          selected={selectedTurn}
          onSelect={(index) => {
            setSelectedTurn(index);
            setSelectedFlowId(null);
          }}
          source={
            <details
              className="source"
              open={transcriptSourceOpen}
              onToggle={(event) => {
                setTranscriptSourceOpen(event.currentTarget.open);
              }}
            >
              <summary>Transcript source</summary>
              <InputBlock label="Transcript" text={transcriptText} onText={onTranscriptText} />
            </details>
          }
        />
      </div>
      <OutputPanel
        evaluation={shown}
        batch={batch}
        graph={graph}
        selectedSourceId={selectedNode?.sourceId ?? null}
        onSelectSource={(sourceId) => {
          const match = graph.nodes.find((node) => node.sourceId === sourceId);
          setSelectedFlowId(match === undefined ? null : match.flowId);
          setSelectedTurn(null);
        }}
      />
    </div>
  );
}

function InputBlock({
  label,
  text,
  onText,
}: {
  label: string;
  text: string;
  onText: (value: string) => void;
}) {
  const fileRef = useRef<HTMLInputElement>(null);
  return (
    <div className="input-block">
      <button
        type="button"
        onClick={() => {
          fileRef.current?.click();
        }}
      >
        Upload {label.toLowerCase()}
      </button>
      <input
        ref={fileRef}
        type="file"
        hidden
        accept=".json,.txt,application/json,text/plain"
        aria-label={`Upload ${label.toLowerCase()}`}
        onChange={(event) => {
          const file = event.target.files?.[0];
          event.target.value = "";
          if (file === undefined) {
            return;
          }
          void file.text().then(onText);
        }}
      />
      <textarea
        aria-label={label}
        value={text}
        spellCheck={false}
        rows={6}
        onChange={(event) => {
          onText(event.target.value);
        }}
      />
    </div>
  );
}

function NodeDetail({ node }: { node: DisplayNode }) {
  return (
    <div className="node-detail">
      <p>
        {node.sourceId} {node.nodeType}
      </p>
      <p>{node.label}</p>
      <p>{node.description}</p>
    </div>
  );
}
