import { useEffect, useState, useCallback, useRef } from 'react';
import DOMPurify from 'dompurify';
import { fileApi } from '@/api/client';
import { useDataStore } from '@/stores/dataStore';
import { useGraphStore } from '@/stores/graphStore';
import { codeToHtml } from 'shiki';
import { LoadingSpinner } from '@/components/shared/LoadingSpinner';

function breadcrumb(filePath: string): string[] {
  return filePath.split('/').filter(Boolean);
}

interface CodeTabProps {
  nodeId: string;
}

export function CodeTab({ nodeId }: CodeTabProps) {
  const fileContent = useDataStore((s) => s.fileContent);
  const setFileContent = useDataStore((s) => s.setFileContent);
  const loading = useDataStore((s) => s.loading);
  const setLoading = useDataStore((s) => s.setLoading);

  const selectedNode = useGraphStore((s) => s.nodes.find((n) => n.id === nodeId));
  const nodeStartLine = selectedNode?.startLine ?? 0;
  const nodeEndLine = selectedNode?.endLine ?? 0;

  const [error, setError] = useState<string | null>(null);
  const [highlightedHtml, setHighlightedHtml] = useState<string>('');
  const scrollRef = useRef<HTMLDivElement>(null);

  const fetchCode = useCallback(
    async (filePath: string | undefined) => {
      if (!filePath) return;
      setLoading('code', true);
      setError(null);
      try {
        const file = await fileApi.getFile(filePath);
        setFileContent(file);
      } catch (err) {
        setError(err instanceof Error ? err.message : 'Failed to load file');
        setFileContent(null);
      } finally {
        setLoading('code', false);
      }
    },
    [setLoading, setFileContent],
  );

  useEffect(() => {
    void fetchCode(selectedNode?.filePath);
  }, [nodeId, selectedNode?.filePath, fetchCode]);

  useEffect(() => {
    if (!fileContent) {
      setHighlightedHtml('');
      return;
    }

    let cancelled = false;

    async function highlight() {
      try {
        const langMap: Record<string, string> = {
          python: 'python',
          py: 'python',
          typescript: 'typescript',
          ts: 'typescript',
          javascript: 'javascript',
          js: 'javascript',
          tsx: 'tsx',
          jsx: 'jsx',
          rust: 'rust',
          go: 'go',
          java: 'java',
          c: 'c',
          cpp: 'cpp',
          csharp: 'csharp',
          ruby: 'ruby',
          shell: 'shell',
          bash: 'bash',
          json: 'json',
          yaml: 'yaml',
          toml: 'toml',
          markdown: 'markdown',
          css: 'css',
          html: 'html',
          sql: 'sql',
        };
        const lang = langMap[fileContent!.language] ?? 'text';

        const html = await codeToHtml(fileContent!.content, {
          lang,
          theme: 'github-dark',
        });

        if (!cancelled) {
          setHighlightedHtml(html);
        }
      } catch {
        // If Shiki fails (unknown lang, etc), fall back to plain text
        if (!cancelled) {
          setHighlightedHtml('');
        }
      }
    }

    void highlight();
    return () => {
      cancelled = true;
    };
  }, [fileContent]);

  useEffect(() => {
    if (!scrollRef.current || !highlightedHtml || nodeStartLine <= 0) return;

    requestAnimationFrame(() => {
      const lineEl = scrollRef.current?.querySelector(
        `[data-line="${nodeStartLine}"]`,
      );
      if (lineEl) {
        lineEl.scrollIntoView({ block: 'center' });
      }
    });
  }, [highlightedHtml, nodeStartLine]);

  if (loading['code']) {
    return (
      <div className="p-4">
        <LoadingSpinner message="Loading source..." />
      </div>
    );
  }

  if (error) {
    return (
      <div className="p-2" style={{ color: 'var(--danger)', fontSize: 11 }}>
        {error}
      </div>
    );
  }

  if (!fileContent) return null;

  const lines = fileContent.content.split('\n');
  const crumbs = breadcrumb(fileContent.path);

  return (
    <div
      style={{ fontFamily: "'JetBrains Mono', monospace" }}
      ref={scrollRef}
    >
      <div
        style={{
          padding: 8,
          borderBottom: '1px solid var(--border)',
          fontSize: 10,
          color: 'var(--text-secondary)',
          display: 'flex',
          flexWrap: 'wrap',
          gap: 2,
        }}
      >
        {crumbs.map((part, i) => (
          <span key={i}>
            {i > 0 && (
              <span style={{ color: 'var(--text-dimmed)', margin: '0 2px' }}>/</span>
            )}
            <span
              style={{
                color: i === crumbs.length - 1 ? 'var(--text-bright)' : 'var(--text-secondary)',
              }}
            >
              {part}
            </span>
          </span>
        ))}
      </div>

      <div style={{ overflow: 'auto' }}>
        {highlightedHtml ? (
          <ShikiRenderedCode
            html={highlightedHtml}
            startLine={nodeStartLine}
            endLine={nodeEndLine}
          />
        ) : (
          <PlainCode
            lines={lines}
            startLine={nodeStartLine}
            endLine={nodeEndLine}
          />
        )}
      </div>
    </div>
  );
}

function ShikiRenderedCode({
  html,
  startLine,
  endLine,
}: {
  html: string;
  startLine: number;
  endLine: number;
}) {
  const lineRegex = /<span class="line">(.*)<\/span>/gm;
  const lineMatches: string[] = [];
  let match: RegExpExecArray | null;
  while ((match = lineRegex.exec(html)) !== null) {
    lineMatches.push(match[1]);
  }

  if (lineMatches.length === 0) {
    return (
      <div
        style={{
          padding: 8,
          fontSize: 11,
          lineHeight: '18px',
          background: 'var(--bg-primary)',
        }}
        dangerouslySetInnerHTML={{ __html: DOMPurify.sanitize(html) }}
      />
    );
  }

  return (
    <table
      style={{
        borderCollapse: 'collapse',
        width: '100%',
        fontSize: 11,
        lineHeight: '18px',
      }}
    >
      <tbody>
        {lineMatches.map((lineHtml, idx) => {
          const lineNum = idx + 1;
          const isHighlighted = lineNum >= startLine && lineNum <= endLine;
          return (
            <tr
              key={lineNum}
              data-line={lineNum}
              style={{
                background: isHighlighted
                  ? 'rgba(57, 211, 83, 0.08)'
                  : 'transparent',
                borderLeft: isHighlighted
                  ? '2px solid var(--accent)'
                  : '2px solid transparent',
              }}
            >
              <td
                style={{
                  fontFamily: "'JetBrains Mono', monospace",
                  color: 'var(--text-dimmed)',
                  textAlign: 'right',
                  paddingRight: 8,
                  paddingLeft: 8,
                  userSelect: 'none',
                  whiteSpace: 'nowrap',
                  width: 1,
                  verticalAlign: 'top',
                }}
              >
                {lineNum}
              </td>
              <td
                style={{
                  fontFamily: "'JetBrains Mono', monospace",
                  paddingRight: 8,
                  whiteSpace: 'pre',
                }}
                dangerouslySetInnerHTML={{ __html: DOMPurify.sanitize(lineHtml) }}
              />
            </tr>
          );
        })}
      </tbody>
    </table>
  );
}

function PlainCode({
  lines,
  startLine,
  endLine,
}: {
  lines: string[];
  startLine: number;
  endLine: number;
}) {
  return (
    <table
      style={{
        borderCollapse: 'collapse',
        width: '100%',
        fontSize: 11,
        lineHeight: '18px',
      }}
    >
      <tbody>
        {lines.map((line, idx) => {
          const lineNum = idx + 1;
          const isHighlighted = lineNum >= startLine && lineNum <= endLine;
          return (
            <tr
              key={lineNum}
              data-line={lineNum}
              style={{
                background: isHighlighted
                  ? 'rgba(57, 211, 83, 0.08)'
                  : 'transparent',
                borderLeft: isHighlighted
                  ? '2px solid var(--accent)'
                  : '2px solid transparent',
              }}
            >
              <td
                style={{
                  fontFamily: "'JetBrains Mono', monospace",
                  color: 'var(--text-dimmed)',
                  textAlign: 'right',
                  paddingRight: 8,
                  paddingLeft: 8,
                  userSelect: 'none',
                  whiteSpace: 'nowrap',
                  width: 1,
                  verticalAlign: 'top',
                }}
              >
                {lineNum}
              </td>
              <td
                style={{
                  fontFamily: "'JetBrains Mono', monospace",
                  color: 'var(--text-primary)',
                  paddingRight: 8,
                  whiteSpace: 'pre',
                }}
              >
                {line}
              </td>
            </tr>
          );
        })}
      </tbody>
    </table>
  );
}
