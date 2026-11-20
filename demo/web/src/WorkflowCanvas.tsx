import { useEffect } from "react";
import {
  Background,
  BackgroundVariant,
  Controls,
  Handle,
  MarkerType,
  Position,
  ReactFlow,
  ReactFlowProvider,
  useReactFlow,
  type Edge,
  type Node,
  type NodeProps,
} from "@xyflow/react";
import "@xyflow/react/dist/style.css";

import { NODE_HEIGHT, NODE_WIDTH, placeNodes, type LayoutMode } from "./layout";
import type { Assessment, DisplayGraph, Evaluation, NodeStatus } from "./model";

export type WorkflowData = {
  sourceId: string;
  label: string;
  description: string;
  nodeType: string;
  status: NodeStatus | null;
  score: number | null;
  highlighted: boolean;
};

export type WorkflowNode = Node<WorkflowData, "workflow">;
export type WorkflowEdge = Edge<{ assessment: Assessment | null }>;

function WorkflowNodeView({ data, selected }: NodeProps<WorkflowNode>) {
  const statusClass = data.status === null ? "status-none" : `status-${data.status}`;
  const highlighted = data.highlighted ? "is-highlighted" : "";
  const current = selected ? "is-selected" : "";
  return (
    <div
      className={`flow-node ${statusClass} ${highlighted} ${current}`}
      role="button"
      tabIndex={0}
      aria-label={`${data.sourceId} ${data.label}`}
    >
      <Handle type="target" position={Position.Top} />
      <div className="node-meta">
        <span>{data.sourceId}</span>
        <span>{data.nodeType}</span>
      </div>
      <div className="node-label">{data.label}</div>
      {data.status !== null ? (
        <div className="node-status">
          {data.status}
          {data.score === null ? "" : ` ${data.score}`}
        </div>
      ) : null}
      <Handle type="source" position={Position.Bottom} />
    </div>
  );
}

const nodeTypes = { workflow: WorkflowNodeView };

function strokeFor(assessment: Assessment | null): string {
  if (assessment === "permitted") {
    return "#067647";
  }
  if (assessment === "invalid") {
    return "#b42318";
  }
  if (assessment === "unresolved") {
    return "#b54708";
  }
  return "#98a2b3";
}

function assessmentFor(graphEdge: { sourceId: string; targetId: string }, evaluation: Evaluation | null) {
  if (evaluation === null || evaluation.kind !== "report") {
    return null;
  }
  for (const hop of evaluation.report.transitions) {
    if (hop.source === graphEdge.sourceId && hop.target === graphEdge.targetId) {
      return hop.assessment;
    }
  }
  return null;
}

export function WorkflowCanvas(props: {
  graph: DisplayGraph;
  evaluation: Evaluation | null;
  layoutMode: LayoutMode;
  highlighted: ReadonlySet<string>;
  onSelect: (flowId: string | null) => void;
}) {
  return (
    <ReactFlowProvider>
      <FlowCanvas {...props} />
    </ReactFlowProvider>
  );
}

function FlowCanvas({
  graph,
  evaluation,
  layoutMode,
  highlighted,
  onSelect,
}: {
  graph: DisplayGraph;
  evaluation: Evaluation | null;
  layoutMode: LayoutMode;
  highlighted: ReadonlySet<string>;
  onSelect: (flowId: string | null) => void;
}) {
  const { fitView } = useReactFlow();
  useEffect(() => {
    const handle = window.setTimeout(() => {
      void fitView({ padding: 0.18, duration: 0, maxZoom: 1 });
    }, 40);
    return () => window.clearTimeout(handle);
  }, [fitView, graph, layoutMode, evaluation]);
  const places = placeNodes(graph.nodes, graph.edges, layoutMode);
  const nodes: WorkflowNode[] = graph.nodes.map((node) => {
    const result =
      evaluation !== null && evaluation.kind === "report"
        ? evaluation.report.node_results[node.sourceId]
        : undefined;
    const place = places.get(node.flowId) ?? { x: 0, y: 0 };
    return {
      id: node.flowId,
      type: "workflow",
      position: place,
      width: NODE_WIDTH,
      height: NODE_HEIGHT,
      data: {
        sourceId: node.sourceId,
        label: node.label,
        description: node.description,
        nodeType: node.nodeType,
        status: result === undefined ? null : result.status,
        score: result === undefined ? null : result.score,
        highlighted: highlighted.has(node.flowId),
      },
      style: { width: NODE_WIDTH, height: NODE_HEIGHT },
    };
  });
  const edges: WorkflowEdge[] = graph.edges.map((edge) => {
    const assessment = assessmentFor(edge, evaluation);
    const stroke = strokeFor(assessment);
    return {
      id: edge.id,
      source: edge.source,
      target: edge.target,
      animated: assessment === "permitted",
      label: assessment ?? "",
      style: {
        stroke,
        strokeWidth: assessment === null ? 1.5 : 2.4,
        strokeDasharray: assessment === "unresolved" ? "6 4" : undefined,
      },
      markerEnd: { type: MarkerType.ArrowClosed, color: stroke },
      data: { assessment },
    };
  });

  return (
    <ReactFlow
      key={`${layoutMode}:${graph.nodes.map((node) => node.flowId).join(",")}:${evaluation === null ? "plain" : "scored"}`}
      nodes={nodes}
      edges={edges}
      nodeTypes={nodeTypes}
      onNodeClick={(_event, node) => {
        onSelect(node.id);
      }}
      onPaneClick={() => {
        onSelect(null);
      }}
      fitView
      fitViewOptions={{ padding: 0.18, maxZoom: 1 }}
      minZoom={0.2}
      nodesDraggable={false}
      nodesConnectable={false}
      elementsSelectable
      proOptions={{ hideAttribution: false }}
      colorMode="light"
    >
      <Background variant={BackgroundVariant.Dots} gap={16} />
      <Controls showInteractive={false} />
    </ReactFlow>
  );
}
