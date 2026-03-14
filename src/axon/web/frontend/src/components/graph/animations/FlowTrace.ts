import type Sigma from 'sigma';
import type { Attributes } from 'graphology-types';
import type { NodeDisplayData, EdgeDisplayData } from 'sigma/types';

const STEP_DURATION_MS = 400;
const ACTIVE_COLOR = '#39d353';  // --accent (green)
const TRAIL_EDGE_COLOR = '#39d353';
const TRAIL_EDGE_SIZE = 3;
const DIM_NODE_COLOR = '#1a2030';
const DIM_EDGE_COLOR = '#0a0e14';

export interface FlowTraceControls {
  pause: () => void;
  resume: () => void;
  stop: () => void;
}

export function startFlowTrace(
  sigma: Sigma,
  stepNodeIds: string[],
  onComplete?: () => void,
): FlowTraceControls {
  const graph = sigma.getGraph();

  const flowNodeSet = new Set(stepNodeIds);

  const visitedNodes = new Set<string>();
  const visitedEdges = new Set<string>();

  let currentStep = 0;
  let paused = false;
  let stopped = false;
  let timer: ReturnType<typeof setTimeout> | null = null;

  const originalNodeReducer = sigma.getSetting('nodeReducer');
  const originalEdgeReducer = sigma.getSetting('edgeReducer');

  const nodeReducer = (
    node: string,
    data: Attributes,
  ): Partial<NodeDisplayData> => {
    const res = { ...data } as Partial<NodeDisplayData>;

    if (visitedNodes.has(node)) {
      res.color = ACTIVE_COLOR;
      res.highlighted = true;
      res.zIndex = 5;
      return res;
    }

    if (flowNodeSet.has(node)) {
      // Upcoming node in the path -- slightly less dim.
      res.color = '#1a3020';
      return res;
    }

    res.color = DIM_NODE_COLOR;
    res.label = null;
    return res;
  };

  const edgeReducer = (
    edge: string,
    data: Attributes,
  ): Partial<EdgeDisplayData> => {
    const res = { ...data } as Partial<EdgeDisplayData>;

    if (visitedEdges.has(edge)) {
      res.color = TRAIL_EDGE_COLOR;
      res.size = TRAIL_EDGE_SIZE;
      return res;
    }

    res.color = DIM_EDGE_COLOR;
    return res;
  };

  sigma.setSetting('nodeReducer', nodeReducer);
  sigma.setSetting('edgeReducer', edgeReducer);
  sigma.refresh();

  function animate(): void {
    if (stopped) return;
    if (paused) return;
    if (currentStep >= stepNodeIds.length) {
      onComplete?.();
      return;
    }

    const nodeId = stepNodeIds[currentStep];
    visitedNodes.add(nodeId);

    if (currentStep > 0) {
      const prevId = stepNodeIds[currentStep - 1];
      // Check both directions since graph is directed: prev->current is the
      // expected flow direction, but we also check current->prev for safety.
      graph.forEachEdge(
        prevId,
        (edge: string, _attrs: Attributes, source: string, target: string) => {
          if (
            (source === prevId && target === nodeId) ||
            (source === nodeId && target === prevId)
          ) {
            visitedEdges.add(edge);
          }
        },
      );
    }

    sigma.refresh();
    currentStep++;

    timer = setTimeout(animate, STEP_DURATION_MS);
  }

  animate();

  return {
    pause: () => {
      paused = true;
      if (timer !== null) {
        clearTimeout(timer);
        timer = null;
      }
    },

    resume: () => {
      if (stopped) return;
      paused = false;
      animate();
    },

    stop: () => {
      stopped = true;
      paused = false;
      if (timer !== null) {
        clearTimeout(timer);
        timer = null;
      }
      sigma.setSetting('nodeReducer', originalNodeReducer);
      sigma.setSetting('edgeReducer', originalEdgeReducer);
      sigma.refresh();
    },
  };
}
