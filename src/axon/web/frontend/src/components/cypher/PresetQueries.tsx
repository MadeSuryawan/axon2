import { useState, useRef, useEffect } from 'react';
import { PRESET_QUERIES } from '@/lib/constants';

interface PresetQueriesProps {
  onSelect: (query: string) => void;
}

export function PresetQueries({ onSelect }: PresetQueriesProps) {
  const [open, setOpen] = useState(false);
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
        Presets &#9662;
      </button>

      {open && (
        <div
          className="absolute left-0 mt-1 w-64 z-50 overflow-hidden"
          style={{
            background: 'var(--bg-elevated)',
            border: '1px solid var(--border)',
            borderRadius: 'var(--radius)',
          }}
        >
          {PRESET_QUERIES.map((preset, i) => (
            <button
              key={i}
              onClick={() => {
                onSelect(preset.query);
                setOpen(false);
              }}
              className="w-full text-left px-3 py-1.5 text-[11px] cursor-pointer bg-transparent border-0 block"
              style={{
                color: 'var(--text-primary)',
                fontFamily: "'JetBrains Mono', monospace",
              }}
              onMouseEnter={(e) => {
                (e.currentTarget as HTMLElement).style.background = 'var(--bg-hover)';
              }}
              onMouseLeave={(e) => {
                (e.currentTarget as HTMLElement).style.background = 'transparent';
              }}
            >
              {preset.label}
            </button>
          ))}
        </div>
      )}
    </div>
  );
}
