import { useState, useCallback } from 'react';
import { Command } from 'cmdk';
import { useViewStore } from '@/stores/viewStore';
import type { ActiveView, LeftTab } from '@/stores/viewStore';
import { useGraphStore } from '@/stores/graphStore';
import { actionApi } from '@/api/client';
import { useSearch } from '@/hooks/useSearch';
import type { SearchResult } from '@/types';

interface PaletteCommand {
  name: string;
  description: string;
  action: () => void;
}

function useCommands(): PaletteCommand[] {
  const leftTab = (tab: LeftTab) => () => useViewStore.getState().setLeftTab(tab);
  const view = (v: ActiveView) => () => useViewStore.getState().setActiveView(v);

  return [
    { name: 'dead-code', description: 'Open Dead Code tab', action: leftTab('dead-code') },
    { name: 'communities', description: 'Open Communities tab', action: leftTab('communities') },
    { name: 'cypher', description: 'Switch to Cypher view', action: view('cypher') },
    { name: 'analysis', description: 'Switch to Analysis view', action: view('analysis') },
    { name: 'explorer', description: 'Switch to Explorer view', action: view('explorer') },
    {
      name: 'reindex',
      description: 'Trigger reindex',
      action: () => { actionApi.reindex(); },
    },
    {
      name: 'hulls',
      description: 'Toggle community hulls',
      action: () => useGraphStore.getState().toggleHulls(),
    },
    {
      name: 'minimap',
      description: 'Toggle minimap',
      action: () => useGraphStore.getState().toggleMinimap(),
    },
  ];
}

function SearchResultItem({ result }: { result: SearchResult }) {
  return (
    <div style={{ display: 'flex', alignItems: 'center', gap: 8, width: '100%' }}>
      <span
        style={{
          fontFamily: 'var(--font-mono, monospace)',
          fontSize: 10,
          padding: '1px 5px',
          borderRadius: 2,
          background: 'var(--bg-hover)',
          color: 'var(--text-secondary)',
          textTransform: 'uppercase',
          flexShrink: 0,
        }}
      >
        {result.label}
      </span>
      <span
        style={{
          fontFamily: 'var(--font-mono, monospace)',
          fontSize: 13,
          color: 'var(--text-primary)',
          overflow: 'hidden',
          textOverflow: 'ellipsis',
          whiteSpace: 'nowrap',
        }}
      >
        {result.name}
      </span>
      <span
        style={{
          marginLeft: 'auto',
          fontFamily: 'var(--font-mono, monospace)',
          fontSize: 11,
          color: 'var(--text-dimmed)',
          flexShrink: 0,
        }}
      >
        {result.filePath}
      </span>
    </div>
  );
}

export function CommandPalette() {
  const open = useViewStore((s) => s.commandPaletteOpen);
  const [inputValue, setInputValue] = useState('');

  const isCommandMode = inputValue.startsWith('>');
  const searchQuery = isCommandMode ? '' : inputValue;
  const { results: searchResults, loading } = useSearch(searchQuery, open && !isCommandMode);
  const commands = useCommands();

  const close = useCallback(() => {
    useViewStore.getState().setCommandPaletteOpen(false);
    setInputValue('');
  }, []);

  const handleSelectSearch = useCallback(
    (value: string) => {
      const result = searchResults.find((r) => r.nodeId === value);
      if (result) {
        useGraphStore.getState().selectNode(result.nodeId);
        useViewStore.getState().setActiveView('explorer');
      }
      close();
    },
    [searchResults, close],
  );

  const handleSelectCommand = useCallback(
    (value: string) => {
      const cmd = commands.find((c) => c.name === value);
      if (cmd) cmd.action();
      close();
    },
    [commands, close],
  );

  if (!open) return null;

  return (
    <div
      style={{
        position: 'fixed',
        inset: 0,
        zIndex: 50,
        display: 'flex',
        alignItems: 'flex-start',
        justifyContent: 'center',
        paddingTop: 120,
        background: 'rgba(0, 0, 0, 0.5)',
        backdropFilter: 'blur(2px)',
      }}
      onClick={close}
    >
      {/* Stop clicks inside the palette from closing the overlay */}
      <div onClick={(e) => e.stopPropagation()}>
        <Command
          label="Command Palette"
          shouldFilter={isCommandMode}
          loop
          style={{
            width: 520,
            background: 'var(--bg-elevated)',
            border: '1px solid var(--border)',
            borderRadius: 2,
            overflow: 'hidden',
            fontFamily: 'var(--font-mono, monospace)',
          }}
        >
          <Command.Input
            value={inputValue}
            onValueChange={setInputValue}
            placeholder={isCommandMode ? 'Run a command...' : 'Search symbols...'}
            autoFocus
            style={{
              width: '100%',
              padding: '10px 14px',
              fontSize: 14,
              fontFamily: 'var(--font-mono, monospace)',
              background: 'var(--bg-elevated)',
              color: 'var(--text-primary)',
              border: 'none',
              borderBottom: '1px solid var(--border)',
              outline: 'none',
            }}
          />

          <Command.List
            style={{
              maxHeight: 320,
              overflowY: 'auto',
              padding: 8,
            }}
          >
            {loading && (
              <Command.Loading
                style={{
                  padding: '8px 14px',
                  fontSize: 12,
                  color: 'var(--text-dimmed)',
                  fontFamily: 'var(--font-mono, monospace)',
                }}
              >
                Searching...
              </Command.Loading>
            )}

            <Command.Empty
              style={{
                padding: '8px 14px',
                fontSize: 12,
                color: 'var(--text-dimmed)',
                fontFamily: 'var(--font-mono, monospace)',
              }}
            >
              No results found.
            </Command.Empty>

            {!isCommandMode && searchResults.length > 0 && (
              <Command.Group heading="Symbols" forceMount>
                {searchResults.map((result) => (
                  <Command.Item
                    key={result.nodeId}
                    value={result.nodeId}
                    keywords={[result.name, result.filePath, result.label]}
                    onSelect={handleSelectSearch}
                    style={{
                      display: 'flex',
                      alignItems: 'center',
                      height: 32,
                      padding: '0 8px',
                      borderRadius: 2,
                      cursor: 'pointer',
                      fontSize: 13,
                      fontFamily: 'var(--font-mono, monospace)',
                      borderLeft: '2px solid transparent',
                    }}
                    data-palette-item=""
                  >
                    <SearchResultItem result={result} />
                  </Command.Item>
                ))}
              </Command.Group>
            )}

            {isCommandMode && (
              <Command.Group heading="Commands" forceMount>
                {commands.map((cmd) => (
                  <Command.Item
                    key={cmd.name}
                    value={cmd.name}
                    keywords={[cmd.description]}
                    onSelect={handleSelectCommand}
                    style={{
                      display: 'flex',
                      alignItems: 'center',
                      height: 32,
                      padding: '0 8px',
                      borderRadius: 2,
                      cursor: 'pointer',
                      fontSize: 13,
                      fontFamily: 'var(--font-mono, monospace)',
                      borderLeft: '2px solid transparent',
                    }}
                    data-palette-item=""
                  >
                    <span style={{ color: 'var(--accent)' }}>{cmd.name}</span>
                    <span
                      style={{
                        marginLeft: 12,
                        color: 'var(--text-dimmed)',
                        fontSize: 12,
                      }}
                    >
                      {cmd.description}
                    </span>
                  </Command.Item>
                ))}
              </Command.Group>
            )}
          </Command.List>
        </Command>
      </div>

      <style>{`
        [data-palette-item][data-selected="true"],
        [data-palette-item][aria-selected="true"] {
          background: var(--bg-hover) !important;
          border-left-color: var(--accent) !important;
        }
        [cmdk-group-heading] {
          font-family: var(--font-mono, monospace);
          font-size: 11px;
          color: var(--text-dimmed);
          padding: 6px 8px 4px;
          text-transform: uppercase;
          letter-spacing: 0.05em;
        }
      `}</style>
    </div>
  );
}
