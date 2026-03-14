import { useState, useRef, useEffect } from 'react';
import { useDataStore } from '@/stores/dataStore';

interface QueryHistoryProps {
  onSelect: (query: string) => void;
}

function relativeTime(ts: number): string {
  const delta = Math.floor((Date.now() - ts) / 1000);
  if (delta < 60) return `${delta}s ago`;
  if (delta < 3600) return `${Math.floor(delta / 60)}m ago`;
  if (delta < 86400) return `${Math.floor(delta / 3600)}h ago`;
  return `${Math.floor(delta / 86400)}d ago`;
}

export function QueryHistory({ onSelect }: QueryHistoryProps) {
  const [open, setOpen] = useState(false);
  const history = useDataStore((s) => s.cypherHistory);
  const ref = useRef<HTMLDivElement>(null);

  useEffect(() => {
    function handleClickOutside(e: MouseEvent) {
      if (ref.current && !ref.current.contains(e.target as Node)) {
        setOpen(false);
      }
    }
    if (open) {
      document.addEventListener('mousedown', handleClickOutside);
    }
    return () => document.removeEventListener('mousedown', handleClickOutside);
  }, [open]);

  return (
    <div ref={ref} className="relative">
      <button
        onClick={() => setOpen((v) => !v)}
        className="px-2 py-1 text-[11px] cursor-pointer bg-transparent"
        style={{
          border: '1px solid var(--border)',
          borderRadius: 'var(--radius)',
          color: 'var(--text-secondary)',
          fontFamily: "'JetBrains Mono', monospace",
        }}
      >
        History &#9662;
      </button>

      {open && (
        <div
          className="absolute left-0 mt-1 w-80 z-50 overflow-y-auto"
          style={{
            maxHeight: 320,
            background: 'var(--bg-elevated)',
            border: '1px solid var(--border)',
            borderRadius: 'var(--radius)',
          }}
        >
          {history.length === 0 ? (
            <div
              className="px-3 py-2 text-[11px]"
              style={{ color: 'var(--text-dimmed)', fontFamily: "'JetBrains Mono', monospace" }}
            >
              No history yet
            </div>
          ) : (
            history.map((entry, i) => (
              <button
                key={i}
                onClick={() => {
                  onSelect(entry.query);
                  setOpen(false);
                }}
                className="w-full text-left px-3 py-1.5 cursor-pointer bg-transparent border-0 flex items-center justify-between gap-2"
                style={{
                  fontFamily: "'JetBrains Mono', monospace",
                }}
                onMouseEnter={(e) => {
                  (e.currentTarget as HTMLElement).style.background = 'var(--bg-hover)';
                }}
                onMouseLeave={(e) => {
                  (e.currentTarget as HTMLElement).style.background = 'transparent';
                }}
              >
                <span
                  className="text-[11px] truncate flex-1 min-w-0"
                  style={{ color: 'var(--text-primary)' }}
                >
                  {entry.query.length > 60
                    ? entry.query.slice(0, 60) + '...'
                    : entry.query}
                </span>
                <span
                  className="text-[10px] shrink-0"
                  style={{ color: 'var(--text-dimmed)' }}
                >
                  {relativeTime(entry.timestamp)}
                </span>
              </button>
            ))
          )}
        </div>
      )}
    </div>
  );
}
