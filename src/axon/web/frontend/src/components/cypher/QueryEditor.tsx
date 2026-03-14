import { useRef, useCallback } from 'react';
import { PresetQueries } from './PresetQueries';
import { QueryHistory } from './QueryHistory';

interface QueryEditorProps {
  value: string;
  onChange: (value: string) => void;
  onExecute: () => void;
  loading: boolean;
}

export function QueryEditor({ value, onChange, onExecute, loading }: QueryEditorProps) {
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  const handleKeyDown = useCallback(
    (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
      if ((e.metaKey || e.ctrlKey) && e.key === 'Enter') {
        e.preventDefault();
        onExecute();
      }
    },
    [onExecute],
  );

  const loadQuery = useCallback(
    (query: string) => {
      onChange(query);
      textareaRef.current?.focus();
    },
    [onChange],
  );

  return (
    <div className="flex flex-col h-full">
      <div className="flex-1 min-h-0">
        <textarea
          ref={textareaRef}
          value={value}
          onChange={(e) => onChange(e.target.value)}
          onKeyDown={handleKeyDown}
          spellCheck={false}
          placeholder="MATCH (n) RETURN n LIMIT 10"
          className="w-full h-full resize-none p-3 outline-none"
          style={{
            background: 'var(--bg-surface)',
            border: '1px solid var(--border)',
            borderRadius: 'var(--radius)',
            color: 'var(--text-primary)',
            fontFamily: "'JetBrains Mono', monospace",
            fontSize: 12,
            lineHeight: 1.6,
          }}
        />
      </div>

      <div
        className="flex items-center gap-2 py-2 shrink-0"
      >
        <button
          onClick={onExecute}
          disabled={loading || !value.trim()}
          className="px-3 py-1 text-[11px] font-medium cursor-pointer border-0"
          style={{
            background: loading || !value.trim() ? 'var(--accent-dim)' : 'var(--accent)',
            color: loading || !value.trim() ? 'var(--text-dimmed)' : 'var(--bg-primary)',
            borderRadius: 'var(--radius)',
            fontFamily: "'JetBrains Mono', monospace",
            opacity: loading || !value.trim() ? 0.6 : 1,
          }}
        >
          {loading ? '...' : '\u25B6 Run'}
        </button>
        <span
          className="text-[10px]"
          style={{ color: 'var(--text-dimmed)', fontFamily: "'JetBrains Mono', monospace" }}
        >
          {'\u2318\u21B5'}
        </span>

        <div className="flex-1" />

        <QueryHistory onSelect={loadQuery} />
        <PresetQueries onSelect={loadQuery} />
      </div>
    </div>
  );
}
