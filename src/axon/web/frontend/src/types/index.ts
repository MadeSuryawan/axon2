export interface GraphNode {
  id: string;
  label: string;
  name: string;
  filePath: string;
  startLine: number;
  endLine: number;
  signature: string;
  language: string;
  className: string;
  isDead: boolean;
  isEntryPoint: boolean;
  isExported: boolean;
}

export interface GraphEdge {
  id: string;
  type: string;
  source: string;
  target: string;
  confidence: number;
  strength?: number;
  stepNumber?: number;
}

export interface OverviewStats {
  nodesByLabel: Record<string, number>;
  edgesByType: Record<string, number>;
  totalNodes: number;
  totalEdges: number;
}

export interface CallerCalleeEntry {
  node: GraphNode;
  confidence: number;
}

export interface NodeContext {
  node: GraphNode;
  callers: CallerCalleeEntry[];
  callees: CallerCalleeEntry[];
  typeRefs: GraphNode[];
  processMemberships: string[];
}

export interface ImpactResult {
  target: GraphNode;
  affected: number;
  depths: Record<string, GraphNode[]>;
}

export interface SearchResult {
  nodeId: string;
  score: number;
  name: string;
  filePath: string;
  label: string;
  snippet: string;
}

export interface HealthBreakdown {
  deadCode: number;
  coupling: number;
  modularity: number;
  confidence: number;
  coverage: number;
}

export interface HealthScore {
  score: number;
  breakdown: HealthBreakdown;
}

export interface DeadCodeEntry {
  name: string;
  type: string;
  line: number;
}

export interface DeadCodeReport {
  total: number;
  byFile: Record<string, DeadCodeEntry[]>;
}

export interface CouplingPair {
  fileA: string;
  fileB: string;
  strength: number;
  coChanges: number;
}

export interface Community {
  id: string;
  name: string;
  memberCount: number;
  cohesion: number | null;
  members: string[];
}

export interface ProcessStep {
  nodeId: string;
  stepNumber: number;
}

export interface Process {
  name: string;
  kind: string | null;
  stepCount: number;
  steps: ProcessStep[];
}

export interface FileContent {
  path: string;
  content: string;
  language: string;
}

export interface FolderNode {
  name: string;
  path: string;
  type: 'folder' | 'file';
  language?: string | null;
  symbolCount?: number;
  children?: FolderNode[];
}

export interface CypherResult {
  columns: string[];
  rows: unknown[][];
  rowCount: number;
  durationMs: number;
}

export interface CypherEntry {
  query: string;
  timestamp: number;
}

export interface ModifiedNodePair {
  before: GraphNode;
  after: GraphNode;
}

export interface DiffResult {
  added: GraphNode[];
  removed: GraphNode[];
  modified: ModifiedNodePair[];
  addedEdges: GraphEdge[];
  removedEdges: GraphEdge[];
}

export interface DiffOverlay {
  added: Set<string>;
  removed: Set<string>;
  modified: Set<string>;
}

export type SSEEvent =
  | { type: 'reindex_start'; data: Record<string, never> }
  | { type: 'reindex_complete'; data: { added?: string[]; removed?: string[]; modified?: string[] } }
  | { type: 'file_changed'; data: { path: string } };

export const NODE_LABELS = [
  'function',
  'class',
  'method',
  'interface',
  'type_alias',
  'enum',
  'file',
  'folder',
  'community',
  'process',
] as const;
export type NodeLabel = (typeof NODE_LABELS)[number];

export const EDGE_TYPES = [
  'calls',
  'imports',
  'extends',
  'implements',
  'uses_type',
  'coupled_with',
  'member_of',
  'step_in_process',
] as const;
export type EdgeType = (typeof EDGE_TYPES)[number];
