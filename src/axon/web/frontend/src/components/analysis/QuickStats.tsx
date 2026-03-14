import type {
  OverviewStats,
  HealthScore,
  DeadCodeReport,
  CouplingPair,
  Community,
  Process,
} from '@/types';

interface QuickStatsProps {
  overview: OverviewStats | null;
  health: HealthScore | null;
  deadCode: DeadCodeReport | null;
  coupling: CouplingPair[];
  communities: Community[];
  processes: Process[];
}

interface StatCellProps {
  label: string;
  value: string | number;
}

function StatCell({ label, value }: StatCellProps) {
  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 2 }}>
      <span
        style={{
          fontSize: 10,
          color: 'var(--text-secondary)',
          fontFamily: "'JetBrains Mono', monospace",
          textTransform: 'uppercase',
          letterSpacing: '0.3px',
        }}
      >
        {label}
      </span>
      <span
        style={{
          fontSize: 14,
          fontWeight: 700,
          color: 'var(--text-bright)',
          fontFamily: "'IBM Plex Mono', monospace",
        }}
      >
        {value}
      </span>
    </div>
  );
}

export function QuickStats({
  overview,
  health,
  deadCode,
  coupling,
  communities,
  processes,
}: QuickStatsProps) {
  const files = overview?.nodesByLabel['file'] ?? 0;
  const totalSymbols = overview?.totalNodes ?? 0;
  const totalEdges = overview?.totalEdges ?? 0;
  const communityCount = communities.length;
  const processCount = processes.length;
  const deadCodeCount = deadCode?.total ?? 0;
  const coupledPairs = coupling.length;

  const avgConfidence =
    health?.breakdown.confidence != null
      ? `${Math.round(health.breakdown.confidence)}%`
      : '--';

  const entryPoints = overview
    ? (overview.nodesByLabel['function'] ?? 0) +
      (overview.nodesByLabel['class'] ?? 0)
    : 0;

  return (
    <div
      style={{
        display: 'grid',
        gridTemplateColumns: 'repeat(auto-fill, minmax(80px, 1fr))',
        gap: 8,
        padding: 6,
      }}
    >
      <StatCell label="Files" value={files} />
      <StatCell label="Symbols" value={totalSymbols} />
      <StatCell label="Relationships" value={totalEdges} />
      <StatCell label="Communities" value={communityCount} />
      <StatCell label="Processes" value={processCount} />
      <StatCell label="Dead Code" value={deadCodeCount} />
      <StatCell label="Coupled Pairs" value={coupledPairs} />
      <StatCell label="Avg Confidence" value={avgConfidence} />
      <StatCell label="Entry Points" value={entryPoints} />
    </div>
  );
}
