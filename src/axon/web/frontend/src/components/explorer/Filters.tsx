import { useGraphStore } from '@/stores/graphStore';

const NODE_TYPE_CONFIG: { type: string; label: string; color: string }[] = [
  { type: 'function', label: 'Function', color: 'var(--node-function)' },
  { type: 'class', label: 'Class', color: 'var(--node-class)' },
  { type: 'method', label: 'Method', color: 'var(--node-method)' },
  { type: 'interface', label: 'Interface', color: 'var(--node-interface)' },
  { type: 'type_alias', label: 'Type Alias', color: 'var(--node-typealias)' },
  { type: 'enum', label: 'Enum', color: 'var(--node-enum)' },
  { type: 'file', label: 'File', color: 'var(--node-file)' },
  { type: 'folder', label: 'Folder', color: 'var(--node-folder)' },
  { type: 'community', label: 'Community', color: 'var(--node-community)' },
  { type: 'process', label: 'Process', color: 'var(--node-process)' },
];

const EDGE_TYPE_CONFIG: { type: string; label: string; color: string; arrow: boolean }[] = [
  { type: 'calls',           label: 'CALLS',        color: '#C8DCFF', arrow: true },
  { type: 'imports',         label: 'IMPORTS',       color: '#B4FFD2', arrow: true },
  { type: 'extends',         label: 'EXTENDS',       color: '#FFD2A0', arrow: true },
  { type: 'implements',      label: 'IMPLEMENTS',    color: '#FFBED2', arrow: true },
  { type: 'uses_type',       label: 'USES_TYPE',     color: '#DCC8FF', arrow: true },
  { type: 'coupled_with',    label: 'COUPLED_WITH',  color: '#FFA0A0', arrow: false },
];

const DEPTH_OPTIONS: { label: string; value: number | null }[] = [
  { label: '1', value: 1 },
  { label: '2', value: 2 },
  { label: '3', value: 3 },
  { label: '5', value: 5 },
  { label: 'All', value: null },
];

const LAYOUT_OPTIONS: { label: string; value: 'force' | 'tree' | 'radial' | 'community' | 'circular' }[] = [
  { label: 'Force', value: 'force' },
  { label: 'Tree', value: 'tree' },
  { label: 'Radial', value: 'radial' },
  { label: 'Community', value: 'community' },
  { label: 'Circular', value: 'circular' },
];

export function Filters() {
  const visibleNodeTypes = useGraphStore((s) => s.visibleNodeTypes);
  const visibleEdgeTypes = useGraphStore((s) => s.visibleEdgeTypes);
  const toggleNodeType = useGraphStore((s) => s.toggleNodeType);
  const toggleEdgeType = useGraphStore((s) => s.toggleEdgeType);
  const depthLimit = useGraphStore((s) => s.depthLimit);
  const setDepthLimit = useGraphStore((s) => s.setDepthLimit);
  const layoutMode = useGraphStore((s) => s.layoutMode);
  const setLayoutMode = useGraphStore((s) => s.setLayoutMode);
  const selectedNodeId = useGraphStore((s) => s.selectedNodeId);

  return (
    <div style={{ padding: 8, display: 'flex', flexDirection: 'column', gap: 12 }}>
      <Section title="Node Types">
        {NODE_TYPE_CONFIG.map((cfg) => (
          <ToggleRow
            key={cfg.type}
            color={cfg.color}
            label={cfg.label}
            active={visibleNodeTypes.has(cfg.type)}
            onToggle={() => toggleNodeType(cfg.type)}
          />
        ))}
      </Section>

      <Section title="Edge Types">
        {EDGE_TYPE_CONFIG.map((cfg) => (
          <ToggleRow
            key={cfg.type}
            label={cfg.label}
            edgeColor={cfg.color}
            edgeArrow={cfg.arrow}
            active={visibleEdgeTypes.has(cfg.type)}
            onToggle={() => toggleEdgeType(cfg.type)}
          />
        ))}
      </Section>

      <Section title="Depth Limit">
        {!selectedNodeId && (
          <div style={{ color: 'var(--text-dimmed)', fontSize: 10, marginBottom: 4 }}>
            Select a node first
          </div>
        )}
        <div style={{ display: 'flex', gap: 4 }}>
          {DEPTH_OPTIONS.map((opt) => (
            <button
              key={opt.label}
              onClick={() => setDepthLimit(opt.value)}
              disabled={!selectedNodeId}
              style={{
                background:
                  depthLimit === opt.value
                    ? 'var(--accent)'
                    : 'var(--bg-elevated)',
                color:
                  depthLimit === opt.value
                    ? 'var(--bg-primary)'
                    : selectedNodeId
                      ? 'var(--text-primary)'
                      : 'var(--text-dimmed)',
                border: 'none',
                borderRadius: 'var(--radius)',
                padding: '2px 8px',
                fontSize: 11,
                fontFamily: "'JetBrains Mono', monospace",
                cursor: selectedNodeId ? 'pointer' : 'default',
                opacity: selectedNodeId ? 1 : 0.5,
              }}
            >
              {opt.label}
            </button>
          ))}
        </div>
      </Section>

      <Section title="Layout">
        <div style={{ display: 'flex', flexWrap: 'wrap', gap: 4 }}>
          {LAYOUT_OPTIONS.map((opt) => {
            const isActive = layoutMode === opt.value;
            return (
              <button
                key={opt.value}
                onClick={() => setLayoutMode(opt.value)}
                style={{
                  background: 'transparent',
                  color: isActive ? 'var(--accent)' : 'var(--text-primary)',
                  border: isActive
                    ? '1px solid var(--accent)'
                    : '1px solid var(--border)',
                  borderRadius: 'var(--radius)',
                  padding: '2px 8px',
                  fontSize: 11,
                  fontFamily: "'JetBrains Mono', monospace",
                  cursor: 'pointer',
                }}
              >
                {opt.label}
              </button>
            );
          })}
        </div>
      </Section>
    </div>
  );
}

function Section({
  title,
  children,
}: {
  title: string;
  children: React.ReactNode;
}) {
  return (
    <div>
      <div
        style={{
          fontSize: 10,
          fontWeight: 600,
          textTransform: 'uppercase',
          letterSpacing: '0.5px',
          color: 'var(--text-secondary)',
          marginBottom: 4,
          fontFamily: "'IBM Plex Mono', monospace",
        }}
      >
        {title}
      </div>
      {children}
    </div>
  );
}

function EdgeIndicator({ color, arrow, active }: { color: string; arrow: boolean; active: boolean }) {
  const strokeColor = active ? color : 'var(--text-dimmed)';
  return (
    <svg width="20" height="10" viewBox="0 0 20 10" style={{ flexShrink: 0 }}>
      <line x1="0" y1="5" x2={arrow ? 14 : 20} y2="5" stroke={strokeColor} strokeWidth="2" />
      {arrow && (
        <polygon points="14,1 20,5 14,9" fill={strokeColor} />
      )}
    </svg>
  );
}

function ToggleRow({
  color,
  edgeColor,
  edgeArrow,
  label,
  active,
  onToggle,
}: {
  color?: string;
  edgeColor?: string;
  edgeArrow?: boolean;
  label: string;
  active: boolean;
  onToggle: () => void;
}) {
  return (
    <div
      onClick={onToggle}
      style={{
        display: 'flex',
        alignItems: 'center',
        gap: 6,
        padding: '4px 8px',
        cursor: 'pointer',
        fontSize: 11,
        borderRadius: 'var(--radius)',
        color: active ? 'var(--text-primary)' : 'var(--text-dimmed)',
      }}
      onMouseEnter={(e) => {
        e.currentTarget.style.background = 'var(--bg-hover)';
      }}
      onMouseLeave={(e) => {
        e.currentTarget.style.background = 'transparent';
      }}
    >
      {/* Node type: colored dot */}
      {color && (
        <span
          style={{
            width: 10,
            height: 10,
            borderRadius: '50%',
            background: active ? color : 'var(--text-dimmed)',
            border: `1.5px solid ${active ? 'rgba(255,255,255,0.15)' : 'transparent'}`,
            flexShrink: 0,
          }}
        />
      )}

      {/* Edge type: colored line/arrow */}
      {edgeColor && (
        <EdgeIndicator color={edgeColor} arrow={edgeArrow ?? false} active={active} />
      )}

      <span style={{ flex: 1 }}>{label}</span>

      <span
        style={{
          width: 12,
          height: 12,
          border: '1px solid',
          borderColor: active ? 'var(--accent)' : 'var(--border)',
          borderRadius: 'var(--radius)',
          background: active ? 'var(--accent)' : 'transparent',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          fontSize: 9,
          color: active ? 'var(--bg-primary)' : 'transparent',
          flexShrink: 0,
          lineHeight: 1,
        }}
      >
        {active ? '\u2713' : ''}
      </span>
    </div>
  );
}
