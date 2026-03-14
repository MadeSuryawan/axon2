import ky from 'ky';
import type {
  GraphNode,
  GraphEdge,
  NodeContext,
  OverviewStats,
  SearchResult,
  ImpactResult,
  DeadCodeReport,
  CouplingPair,
  Community,
  Process,
  HealthScore,
  FolderNode,
  FileContent,
  CypherResult,
  DiffResult,
} from '@/types';

const api = ky.create({
  prefixUrl: '/api',
  timeout: 30_000,
});

export const graphApi = {
  getGraph: () =>
    api.get('graph').json<{ nodes: GraphNode[]; edges: GraphEdge[] }>(),

  getNode: (id: string) =>
    api.get(`node/${id}`).json<NodeContext>(),

  getOverview: () =>
    api.get('overview').json<OverviewStats>(),
};

export const searchApi = {
  search: (query: string, limit = 20, options?: { signal?: AbortSignal }) =>
    api
      .post('search', { json: { query, limit }, signal: options?.signal })
      .json<{ results: SearchResult[] }>(),
};

export const analysisApi = {
  getImpact: (nodeId: string, depth = 3) =>
    api
      .get(`impact/${nodeId}`, { searchParams: { depth } })
      .json<ImpactResult>(),

  getDeadCode: () =>
    api.get('dead-code').json<DeadCodeReport>(),

  getCoupling: () =>
    api.get('coupling').json<{ pairs: CouplingPair[] }>(),

  getCommunities: () =>
    api.get('communities').json<{ communities: Community[] }>(),

  getProcesses: () =>
    api.get('processes').json<{ processes: Process[] }>(),

  getHealth: () =>
    api.get('health').json<HealthScore>(),
};

export const fileApi = {
  getTree: () =>
    api.get('tree').json<{ tree: FolderNode[] }>(),

  getFile: (path: string) =>
    api.get('file', { searchParams: { path } }).json<FileContent>(),
};

export const cypherApi = {
  execute: (query: string) =>
    api.post('cypher', { json: { query } }).json<CypherResult>(),
};

export const diffApi = {
  compare: (base: string, compare: string) =>
    api
      .post('diff', { json: { base, compare } })
      .json<DiffResult>(),
};

export const actionApi = {
  reindex: () =>
    api.post('reindex').json<{ status: string }>(),
};
