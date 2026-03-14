import { FolderTree, SlidersHorizontal, Grid3X3, Skull } from 'lucide-react';
import { useViewStore, type LeftTab } from '@/stores/viewStore';
import { FileTree } from './FileTree';
import { Filters } from './Filters';
import { Communities } from './Communities';
import { DeadCode } from './DeadCode';

const TABS: { id: LeftTab; icon: typeof FolderTree; title: string }[] = [
  { id: 'files', icon: FolderTree, title: 'Files' },
  { id: 'filters', icon: SlidersHorizontal, title: 'Filters' },
  { id: 'communities', icon: Grid3X3, title: 'Communities' },
  { id: 'dead-code', icon: Skull, title: 'Dead Code' },
];

export function ExplorerSidebar() {
  const activeTab = useViewStore((s) => s.leftSidebarTab);
  const setLeftTab = useViewStore((s) => s.setLeftTab);

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
          const Icon = tab.icon;
          const isActive = activeTab === tab.id;
          return (
            <button
              key={tab.id}
              onClick={() => setLeftTab(tab.id)}
              title={tab.title}
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
              }}
            >
              <Icon size={16} />
            </button>
          );
        })}
      </div>

      <div className="flex-1 overflow-y-auto overflow-x-hidden">
        {activeTab === 'files' && <FileTree />}
        {activeTab === 'filters' && <Filters />}
        {activeTab === 'communities' && <Communities />}
        {activeTab === 'dead-code' && <DeadCode />}
      </div>
    </div>
  );
}
