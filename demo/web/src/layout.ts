import dagre from "@dagrejs/dagre";

import type { DisplayEdge, DisplayNode, Point } from "./model";

export const NODE_WIDTH = 248;
export const NODE_HEIGHT = 120;

export type LayoutMode = "ranked" | "stored";

export function placeNodes(
  nodes: readonly DisplayNode[],
  edges: readonly DisplayEdge[],
  mode: LayoutMode,
): Map<string, Point> {
  if (mode === "stored" && nodes.some((node) => node.position !== null)) {
    return storedPlaces(nodes);
  }
  return rankedPlaces(nodes, edges);
}

function storedPlaces(nodes: readonly DisplayNode[]): Map<string, Point> {
  const placed = new Map<string, Point>();
  let fallback = 0;
  for (const node of nodes) {
    if (node.position !== null) {
      placed.set(node.flowId, node.position);
      continue;
    }
    placed.set(node.flowId, { x: 0, y: fallback });
    fallback += NODE_HEIGHT + 40;
  }
  return placed;
}

function rankedPlaces(
  nodes: readonly DisplayNode[],
  edges: readonly DisplayEdge[],
): Map<string, Point> {
  const graph = new dagre.graphlib.Graph({ multigraph: true });
  graph.setDefaultEdgeLabel(() => ({}));
  graph.setGraph({ rankdir: "TB", nodesep: 56, ranksep: 72 });
  const known = new Set<string>();
  for (const node of nodes) {
    known.add(node.flowId);
    graph.setNode(node.flowId, { width: NODE_WIDTH, height: NODE_HEIGHT });
  }
  for (const edge of edges) {
    if (!known.has(edge.source) || !known.has(edge.target)) {
      continue;
    }
    graph.setEdge(edge.source, edge.target, {}, edge.id);
  }
  dagre.layout(graph);
  const placed = new Map<string, Point>();
  nodes.forEach((node, index) => {
    const label: { x?: number; y?: number } = graph.node(node.flowId);
    if (
      typeof label.x === "number" &&
      Number.isFinite(label.x) &&
      typeof label.y === "number" &&
      Number.isFinite(label.y)
    ) {
      placed.set(node.flowId, {
        x: label.x - NODE_WIDTH / 2,
        y: label.y - NODE_HEIGHT / 2,
      });
      return;
    }
    placed.set(node.flowId, { x: 0, y: index * (NODE_HEIGHT + 40) });
  });
  return placed;
}
