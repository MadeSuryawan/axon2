import { Header } from '@/components/layout/Header';
import { StatusBar } from '@/components/layout/StatusBar';
import { PanelLayout } from '@/components/layout/PanelLayout';
import { ExplorerSidebar } from '@/components/explorer/ExplorerSidebar';
import { GraphCanvas } from '@/components/graph/GraphCanvas';
import { DetailPanel } from '@/components/detail/DetailPanel';
import { AnalysisView } from '@/components/analysis/AnalysisView';
import { CypherView } from '@/components/cypher/CypherView';
import { CommandPalette } from '@/components/shared/CommandPalette';
import { ErrorBoundary } from '@/components/shared/ErrorBoundary';
import { useKeyboard } from '@/hooks/useKeyboard';
import { useViewStore } from '@/stores/viewStore';
import { useSSE } from '@/hooks/useSSE';

export function App() {
  const activeView = useViewStore((s) => s.activeView);
  useKeyboard();
  useSSE();

  return (
    <div
      className="h-screen w-screen flex flex-col overflow-hidden"
      style={{ background: 'var(--bg-primary)' }}
    >
      <Header />
      <main className="flex-1 overflow-hidden">
        <ErrorBoundary>
          {activeView === 'explorer' && (
            <PanelLayout
              left={<ExplorerSidebar />}
              center={<GraphCanvas />}
              right={<DetailPanel />}
            />
          )}
          {activeView === 'analysis' && <AnalysisView />}
          {activeView === 'cypher' && <CypherView />}
        </ErrorBoundary>
      </main>
      <StatusBar />
      <CommandPalette />
    </div>
  );
}
