import { create } from 'zustand';

export type ActiveView = 'explorer' | 'analysis' | 'cypher';
export type LeftTab = 'files' | 'filters' | 'communities' | 'dead-code';
export type RightTab = 'context' | 'impact' | 'code' | 'processes';

interface ViewStore {
  activeView: ActiveView;
  leftSidebarOpen: boolean;
  leftSidebarTab: LeftTab;
  rightPanelOpen: boolean;
  rightPanelTab: RightTab;
  commandPaletteOpen: boolean;

  setActiveView: (view: ActiveView) => void;
  toggleLeftSidebar: () => void;
  setLeftSidebarOpen: (open: boolean) => void;
  toggleRightPanel: () => void;
  setRightPanelOpen: (open: boolean) => void;
  setLeftTab: (tab: LeftTab) => void;
  setRightTab: (tab: RightTab) => void;
  toggleCommandPalette: () => void;
  setCommandPaletteOpen: (open: boolean) => void;
}

export const useViewStore = create<ViewStore>((set) => ({
  activeView: 'explorer',
  leftSidebarOpen: true,
  leftSidebarTab: 'files',
  rightPanelOpen: true,
  rightPanelTab: 'context',
  commandPaletteOpen: false,

  setActiveView: (view) => set({ activeView: view }),
  toggleLeftSidebar: () => set((s) => ({ leftSidebarOpen: !s.leftSidebarOpen })),
  setLeftSidebarOpen: (open) => set({ leftSidebarOpen: open }),
  toggleRightPanel: () => set((s) => ({ rightPanelOpen: !s.rightPanelOpen })),
  setRightPanelOpen: (open) => set({ rightPanelOpen: open }),
  setLeftTab: (tab) => set({ leftSidebarTab: tab, leftSidebarOpen: true }),
  setRightTab: (tab) => set({ rightPanelTab: tab, rightPanelOpen: true }),
  toggleCommandPalette: () => set((s) => ({ commandPaletteOpen: !s.commandPaletteOpen })),
  setCommandPaletteOpen: (open) => set({ commandPaletteOpen: open }),
}));
