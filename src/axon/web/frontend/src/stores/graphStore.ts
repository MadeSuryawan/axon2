import { create } from 'zustand';
import type { GraphNode, GraphEdge, Community, OverviewStats, DiffOverlay } from '@/types';

interface GraphStore {
  nodes: GraphNode[];
  edges: GraphEdge[];
  communities: Community[];
  overview: OverviewStats | null;
  selectedNodeId: string | null;
  hoveredNodeId: string | null;
  highlightedNodeIds: Set<string>;
  blastRadiusNodes: Map<string, number>;
  flowTraceNodeIds: string[];
  diffOverlay: DiffOverlay | null;
  visibleNodeTypes: Set<string>;
  visibleEdgeTypes: Set<string>;
  depthLimit: number | null;
  layoutMode: 'force' | 'tree' | 'radial' | 'community' | 'circular';
  hullsVisible: boolean;
  minimapVisible: boolean;

  setGraphData: (nodes: GraphNode[], edges: GraphEdge[]) => void;
  setCommunities: (communities: Community[]) => void;
  setOverview: (overview: OverviewStats) => void;
  selectNode: (id: string | null) => void;
  setHoveredNode: (id: string | null) => void;
  setHighlightedNodes: (ids: Set<string>) => void;
  setBlastRadius: (nodes: Map<string, number>) => void;
  clearBlastRadius: () => void;
  setFlowTrace: (nodeIds: string[]) => void;
  clearFlowTrace: () => void;
  setDiffOverlay: (overlay: DiffOverlay | null) => void;
  toggleNodeType: (type: string) => void;
  toggleEdgeType: (type: string) => void;
  setDepthLimit: (depth: number | null) => void;
  setLayoutMode: (mode: 'force' | 'tree' | 'radial' | 'community' | 'circular') => void;
  toggleHulls: () => void;
  toggleMinimap: () => void;
}

export const useGraphStore = create<GraphStore>((set) => ({
  nodes: [],
  edges: [],
  communities: [],
  overview: null,
  selectedNodeId: null,
  hoveredNodeId: null,
  highlightedNodeIds: new Set(),
  blastRadiusNodes: new Map(),
  flowTraceNodeIds: [],
  diffOverlay: null,
  visibleNodeTypes: new Set(['function', 'class', 'method', 'interface']),
  visibleEdgeTypes: new Set(['calls']),
  depthLimit: null,
  layoutMode: 'force',
  hullsVisible: false,
  minimapVisible: false,

  setGraphData: (nodes, edges) => set({ nodes, edges }),
  setCommunities: (communities) => set({ communities }),
  setOverview: (overview) => set({ overview }),
  selectNode: (id) => set({ selectedNodeId: id }),
  setHoveredNode: (id) => set({ hoveredNodeId: id }),
  setHighlightedNodes: (ids) => set({ highlightedNodeIds: ids }),
  setBlastRadius: (nodes) => set({ blastRadiusNodes: nodes }),
  clearBlastRadius: () => set({ blastRadiusNodes: new Map(), highlightedNodeIds: new Set() }),
  setFlowTrace: (nodeIds) => set({ flowTraceNodeIds: nodeIds }),
  clearFlowTrace: () => set({ flowTraceNodeIds: [], highlightedNodeIds: new Set() }),
  setDiffOverlay: (overlay) => set({ diffOverlay: overlay }),
  toggleNodeType: (type) => set((s) => {
    const next = new Set(s.visibleNodeTypes);
    next.has(type) ? next.delete(type) : next.add(type);
    return { visibleNodeTypes: next };
  }),
  toggleEdgeType: (type) => set((s) => {
    const next = new Set(s.visibleEdgeTypes);
    const enabling = !next.has(type);
    enabling ? next.add(type) : next.delete(type);

    // Auto-enable node types required for edges to be visible.
    // Sigma hides edges when their endpoint nodes are hidden.
    const EDGE_REQUIRES_NODES: Record<string, string[]> = {
      imports: ['file'],
      coupled_with: ['file'],
      step_in_process: ['process'],
    };
    if (enabling && EDGE_REQUIRES_NODES[type]) {
      const nodeTypes = new Set(s.visibleNodeTypes);
      for (const nt of EDGE_REQUIRES_NODES[type]) {
        nodeTypes.add(nt);
      }
      return { visibleEdgeTypes: next, visibleNodeTypes: nodeTypes };
    }

    return { visibleEdgeTypes: next };
  }),
  setDepthLimit: (depth) => set({ depthLimit: depth }),
  setLayoutMode: (mode) => set({ layoutMode: mode }),
  toggleHulls: () => set((s) => ({ hullsVisible: !s.hullsVisible })),
  toggleMinimap: () => set((s) => ({ minimapVisible: !s.minimapVisible })),
}));
