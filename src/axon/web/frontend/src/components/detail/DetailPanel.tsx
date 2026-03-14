import { useViewStore, type RightTab } from '@/stores/viewStore';
import { useGraphStore } from '@/stores/graphStore';
import { ContextTab } from './ContextTab';
import { ImpactTab } from './ImpactTab';
import { CodeTab } from './CodeTab';
import { ProcessesTab } from './ProcessesTab';
import { EmptyState } from '@/components/shared/EmptyState';

const TABS: { id: RightTab; label: string }[] = [
  { id: 'context', label: 'Context' },
  { id: 'impact', label: 'Impact' },
  { id: 'code', label: 'Code' },
  { id: 'processes', label: 'Processes' },
];

export function DetailPanel() {
  const activeTab = useViewStore((s) => s.rightPanelTab);
  const setRightTab = useViewStore((s) => s.setRightTab);
  const selectedNodeId = useGraphStore((s) => s.selectedNodeId);

  return (
    <div className="flex flex-col h-full">
      <div
        className="flex shrink-0"
        style={{
          height: 32,
          background: 'var(--bg-surface)',
          borderBottom: '1px solid var(--border)',
        }}
      >
        {TABS.map((tab) => {
          const isActive = activeTab === tab.id;
          return (
            <button
              key={tab.id}
              onClick={() => setRightTab(tab.id)}
              className="flex items-center justify-center flex-1"
              style={{
                background: 'transparent',
                border: 'none',
                borderBottom: isActive
                  ? '2px solid var(--accent)'
                  : '2px solid transparent',
                color: isActive ? 'var(--text-bright)' : 'var(--text-secondary)',
                cursor: 'pointer',
                padding: 0,
                fontFamily: "'JetBrains Mono', monospace",
                fontSize: 11,
                fontWeight: 500,
                letterSpacing: '0.3px',
                textTransform: 'uppercase',
              }}
            >
              {tab.label}
            </button>
          );
        })}
      </div>

      <div className="flex-1 overflow-y-auto overflow-x-hidden">
        {!selectedNodeId ? (
          <EmptyState message="Select a node on the graph" />
        ) : (
          <>
            {activeTab === 'context' && <ContextTab nodeId={selectedNodeId} />}
            {activeTab === 'impact' && <ImpactTab nodeId={selectedNodeId} />}
            {activeTab === 'code' && <CodeTab nodeId={selectedNodeId} />}
            {activeTab === 'processes' && <ProcessesTab nodeId={selectedNodeId} />}
          </>
        )}
      </div>
    </div>
  );
}
