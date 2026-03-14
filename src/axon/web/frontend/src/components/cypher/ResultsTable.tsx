import { useState, useEffect, useMemo, useCallback } from 'react';
import { RESULTS_PAGE_SIZE } from '@/lib/constants';
import type { CypherResult } from '@/types';
import { EmptyState } from '@/components/shared/EmptyState';

interface ResultsTableProps {
  result: CypherResult | null;
  error: string | null;
}

type SortDirection = 'asc' | 'desc';

interface SortState {
  column: number;
  direction: SortDirection;
}

function formatCell(value: unknown): string {
  if (value === null || value === undefined) return '';
  if (typeof value === 'object') return JSON.stringify(value);
  return String(value);
}

function compareValues(a: unknown, b: unknown): number {
  if (a === null || a === undefined) return -1;
  if (b === null || b === undefined) return 1;
  if (typeof a === 'number' && typeof b === 'number') return a - b;
  return String(a).localeCompare(String(b));
}

export function ResultsTable({ result, error }: ResultsTableProps) {
  const [sort, setSort] = useState<SortState | null>(null);
  const [page, setPage] = useState(0);

  useEffect(() => {
    setPage(0);
  }, [result]);

  const sortedRows = useMemo(() => {
    if (!result) return [];
    const rows = [...result.rows];
    if (sort) {
      rows.sort((a, b) => {
        const cmp = compareValues(a[sort.column], b[sort.column]);
        return sort.direction === 'asc' ? cmp : -cmp;
      });
    }
    return rows;
  }, [result, sort]);

  const totalPages = result ? Math.max(1, Math.ceil(sortedRows.length / RESULTS_PAGE_SIZE)) : 1;
  const pagedRows = sortedRows.slice(
    page * RESULTS_PAGE_SIZE,
    (page + 1) * RESULTS_PAGE_SIZE,
  );

  const handleSort = useCallback((colIdx: number) => {
    setSort((prev) => {
      if (prev?.column === colIdx) {
        return prev.direction === 'asc'
          ? { column: colIdx, direction: 'desc' }
          : null;
      }
      return { column: colIdx, direction: 'asc' };
    });
    setPage(0);
  }, []);

  const copyCSV = useCallback(() => {
    if (!result) return;
    const header = result.columns.join(',');
    const rows = result.rows.map((row) =>
      row.map((cell) => {
        const s = formatCell(cell);
        return s.includes(',') || s.includes('"')
          ? `"${s.replace(/"/g, '""')}"`
          : s;
      }).join(','),
    );
    navigator.clipboard.writeText([header, ...rows].join('\n'));
  }, [result]);

  const copyJSON = useCallback(() => {
    if (!result) return;
    const data = result.rows.map((row) => {
      const obj: Record<string, unknown> = {};
      result.columns.forEach((col, i) => {
        obj[col] = row[i];
      });
      return obj;
    });
    navigator.clipboard.writeText(JSON.stringify(data, null, 2));
  }, [result]);

  if (error) {
    return (
      <div className="flex flex-col h-full">
        <div
          className="p-3 text-[11px]"
          style={{
            color: 'var(--danger)',
            fontFamily: "'JetBrains Mono', monospace",
            whiteSpace: 'pre-wrap',
          }}
        >
          {error}
        </div>
      </div>
    );
  }

  if (!result) {
    return <EmptyState message="Run a query to see results" />;
  }

  return (
    <div className="flex flex-col h-full">
      <div className="flex-1 min-h-0 overflow-auto">
        <table
          className="w-full"
          style={{
            borderCollapse: 'collapse',
            fontFamily: "'JetBrains Mono', monospace",
            fontSize: 11,
          }}
        >
          <thead>
            <tr>
              {result.columns.map((col, i) => (
                <th
                  key={i}
                  onClick={() => handleSort(i)}
                  className="text-left px-3 py-1.5 cursor-pointer select-none whitespace-nowrap"
                  style={{
                    position: 'sticky',
                    top: 0,
                    background: 'var(--bg-elevated)',
                    color: 'var(--text-bright)',
                    borderBottom: '1px solid var(--border)',
                    fontWeight: 600,
                    fontSize: 10,
                    textTransform: 'uppercase',
                    letterSpacing: '0.5px',
                  }}
                >
                  {col}
                  {sort?.column === i && (
                    <span style={{ marginLeft: 4, color: 'var(--accent)' }}>
                      {sort.direction === 'asc' ? '\u25B2' : '\u25BC'}
                    </span>
                  )}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {pagedRows.map((row, rowIdx) => (
              <tr
                key={rowIdx}
                onMouseEnter={(e) => {
                  (e.currentTarget as HTMLElement).style.background = 'var(--bg-hover)';
                }}
                onMouseLeave={(e) => {
                  (e.currentTarget as HTMLElement).style.background = 'transparent';
                }}
              >
                {row.map((cell, cellIdx) => (
                  <td
                    key={cellIdx}
                    className="px-3 py-1 whitespace-nowrap"
                    style={{
                      color: 'var(--text-primary)',
                      borderBottom: '1px solid var(--border-muted)',
                      maxWidth: 300,
                      overflow: 'hidden',
                      textOverflow: 'ellipsis',
                    }}
                  >
                    {formatCell(cell)}
                  </td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      <div
        className="flex items-center justify-between px-3 py-1.5 shrink-0"
        style={{
          borderTop: '1px solid var(--border)',
          background: 'var(--bg-surface)',
        }}
      >
        <div className="flex items-center gap-3">
          <span
            className="text-[10px]"
            style={{ color: 'var(--text-secondary)', fontFamily: "'JetBrains Mono', monospace" }}
          >
            {result.rowCount} rows &middot; {result.durationMs}ms
          </span>

          {totalPages > 1 && (
            <div className="flex items-center gap-1">
              <button
                onClick={() => setPage((p) => Math.max(0, p - 1))}
                disabled={page === 0}
                className="px-1.5 py-0.5 text-[10px] cursor-pointer bg-transparent border-0"
                style={{
                  color: page === 0 ? 'var(--text-dimmed)' : 'var(--text-secondary)',
                  fontFamily: "'JetBrains Mono', monospace",
                }}
              >
                &lt;
              </button>
              <span
                className="text-[10px]"
                style={{ color: 'var(--text-secondary)', fontFamily: "'JetBrains Mono', monospace" }}
              >
                {page + 1}/{totalPages}
              </span>
              <button
                onClick={() => setPage((p) => Math.min(totalPages - 1, p + 1))}
                disabled={page >= totalPages - 1}
                className="px-1.5 py-0.5 text-[10px] cursor-pointer bg-transparent border-0"
                style={{
                  color: page >= totalPages - 1 ? 'var(--text-dimmed)' : 'var(--text-secondary)',
                  fontFamily: "'JetBrains Mono', monospace",
                }}
              >
                &gt;
              </button>
            </div>
          )}
        </div>

        <div className="flex items-center gap-2">
          <button
            onClick={copyCSV}
            className="px-2 py-0.5 text-[10px] cursor-pointer bg-transparent"
            style={{
              border: '1px solid var(--border)',
              borderRadius: 'var(--radius)',
              color: 'var(--text-secondary)',
              fontFamily: "'JetBrains Mono', monospace",
            }}
          >
            Copy CSV
          </button>
          <button
            onClick={copyJSON}
            className="px-2 py-0.5 text-[10px] cursor-pointer bg-transparent"
            style={{
              border: '1px solid var(--border)',
              borderRadius: 'var(--radius)',
              color: 'var(--text-secondary)',
              fontFamily: "'JetBrains Mono', monospace",
            }}
          >
            Copy JSON
          </button>
        </div>
      </div>
    </div>
  );
}
