import { create } from 'zustand';
import type {
  CypherEntry,
  CypherResult,
  DeadCodeReport,
  HealthScore,
  ImpactResult,
  NodeContext,
  Process,
} from '@/types';

interface DataStore {
  nodeContext: NodeContext | null;
  impactResult: ImpactResult | null;
  fileContent: { path: string; content: string; language: string } | null;
  healthScore: HealthScore | null;
  deadCode: DeadCodeReport | null;
  allProcesses: Process[] | null;
  cypherHistory: CypherEntry[];
  cypherResult: CypherResult | null;
  loading: Record<string, boolean>;

  setNodeContext: (ctx: NodeContext | null) => void;
  setImpactResult: (result: ImpactResult | null) => void;
  setFileContent: (content: { path: string; content: string; language: string } | null) => void;
  setHealthScore: (score: HealthScore | null) => void;
  setDeadCode: (report: DeadCodeReport | null) => void;
  setAllProcesses: (processes: Process[] | null) => void;
  setCypherResult: (result: CypherResult | null) => void;
  addCypherHistory: (query: string) => void;
  setLoading: (key: string, value: boolean) => void;
}

export const useDataStore = create<DataStore>((set) => ({
  nodeContext: null,
  impactResult: null,
  fileContent: null,
  healthScore: null,
  deadCode: null,
  allProcesses: null,
  cypherHistory: (() => {
    try { return JSON.parse(localStorage.getItem('axon-cypher-history') || '[]'); }
    catch { return []; }
  })(),
  cypherResult: null,
  loading: {},

  setNodeContext: (ctx) => set({ nodeContext: ctx }),
  setImpactResult: (result) => set({ impactResult: result }),
  setFileContent: (content) => set({ fileContent: content }),
  setHealthScore: (score) => set({ healthScore: score }),
  setDeadCode: (report) => set({ deadCode: report }),
  setAllProcesses: (processes) => set({ allProcesses: processes }),
  setCypherResult: (result) => set({ cypherResult: result }),
  addCypherHistory: (query) => set((s) => {
    const entry: CypherEntry = { query, timestamp: Date.now() };
    const history = [entry, ...s.cypherHistory].slice(0, 20);
    localStorage.setItem('axon-cypher-history', JSON.stringify(history));
    return { cypherHistory: history };
  }),
  setLoading: (key, value) => set((s) => ({
    loading: { ...s.loading, [key]: value },
  })),
}));
