import { useMemo, useState } from 'react';
import { useGraphStore } from '@/stores/graphStore';
import { shortPath } from '@/lib/utils';

interface TreeNode {
  id: string;
  name: string;
  filePath: string;
  startLine: number;
  label: string;
  isInterface: boolean;
  edgeType: 'extends' | 'implements' | 'root';
  children: TreeNode[];
}

export function InheritanceTree() {
  const nodes = useGraphStore((s) => s.nodes);
  const edges = useGraphStore((s) => s.edges);

  const roots = useMemo(() => buildForest(nodes, edges), [nodes, edges]);

  const methodCounts = useMemo(() => {
    const map = new Map<string, number>();
    for (const n of nodes) {
      if (n.label === 'method' && n.className) {
        map.set(n.className, (map.get(n.className) ?? 0) + 1);
      }
    }
    return map;
  }, [nodes]);

  if (roots.length === 0) {
    return (
      <div
        style={{
          padding: 16,
          textAlign: 'center',
          fontSize: 12,
          color: 'var(--text-dimmed)',
          fontFamily: "'JetBrains Mono', monospace",
        }}
      >
        No inheritance hierarchy found
      </div>
    );
  }

  return (
    <div style={{ padding: 8, display: 'flex', flexDirection: 'column', gap: 4 }}>
      {roots.map((root) => (
        <TreeNodeRow key={root.id} node={root} depth={0} methodCounts={methodCounts} />
      ))}
    </div>
  );
}

function TreeNodeRow({ node, depth, methodCounts }: { node: TreeNode; depth: number; methodCounts: Map<string, number> }) {
  const [expanded, setExpanded] = useState(true);
  const hasChildren = node.children.length > 0;

  const methodCount = methodCounts.get(node.name) ?? 0;

  const isInterface = node.isInterface;
  const isDashed = node.edgeType === 'implements';

  return (
    <div>
      <div
        style={{
          display: 'flex',
          alignItems: 'center',
          gap: 4,
          paddingLeft: depth * 16,
          paddingTop: 2,
          paddingBottom: 2,
          cursor: hasChildren ? 'pointer' : 'default',
          userSelect: 'none',
        }}
        onClick={() => hasChildren && setExpanded(!expanded)}
        onMouseEnter={(e) => {
          e.currentTarget.style.background = 'var(--bg-hover)';
        }}
        onMouseLeave={(e) => {
          e.currentTarget.style.background = 'transparent';
        }}
      >
        {/* Connector line indicator */}
        {depth > 0 && (
          <span
            style={{
              width: 12,
              borderBottom: isDashed
                ? '1px dashed var(--text-dimmed)'
                : '1px solid var(--text-dimmed)',
              height: 0,
              flexShrink: 0,
            }}
          />
        )}

        {/* Expand/collapse */}
        {hasChildren ? (
          <span
            style={{
              width: 12,
              textAlign: 'center',
              fontSize: 10,
              color: 'var(--text-dimmed)',
              flexShrink: 0,
            }}
          >
            {expanded ? '\u25BE' : '\u25B8'}
          </span>
        ) : (
          <span style={{ width: 12, flexShrink: 0 }} />
        )}

        {/* Class/interface name */}
        <span
          style={{
            fontSize: 11,
            fontFamily: "'JetBrains Mono', monospace",
            fontWeight: 600,
            color: isInterface ? 'var(--cyan)' : 'var(--node-class)',
            fontStyle: isInterface ? 'italic' : 'normal',
          }}
        >
          {node.name}
        </span>

        {/* File:line */}
        <span
          style={{
            fontSize: 9,
            color: 'var(--text-dimmed)',
            fontFamily: "'JetBrains Mono', monospace",
          }}
        >
          {shortPath(node.filePath)}:{node.startLine}
        </span>

        {/* Method count badge */}
        {methodCount > 0 && (
          <span
            style={{
              fontSize: 9,
              fontWeight: 600,
              fontFamily: "'JetBrains Mono', monospace",
              color: 'var(--text-secondary)',
              background: 'var(--bg-primary)',
              padding: '0 4px',
              borderRadius: 'var(--radius)',
            }}
          >
            {methodCount}m
          </span>
        )}
      </div>

      {/* Children */}
      {expanded &&
        node.children.map((child) => (
          <TreeNodeRow key={child.id} node={child} depth={depth + 1} methodCounts={methodCounts} />
        ))}
    </div>
  );
}

function buildForest(
  nodes: { id: string; name: string; filePath: string; startLine: number; label: string }[],
  edges: { type: string; source: string; target: string }[],
): TreeNode[] {
  const heritageEdges = edges.filter(
    (e) => e.type === 'extends' || e.type === 'implements',
  );

  if (heritageEdges.length === 0) return [];

  const nodeMap = new Map<string, (typeof nodes)[number]>();
  for (const n of nodes) {
    nodeMap.set(n.id, n);
  }

  // Edge direction: child --extends/implements--> parent (source=child, target=parent)
  const childrenMap = new Map<string, { childId: string; edgeType: 'extends' | 'implements' }[]>();
  const hasParent = new Set<string>();

  for (const e of heritageEdges) {
    const parentId = e.target;
    const childId = e.source;
    hasParent.add(childId);

    if (!childrenMap.has(parentId)) {
      childrenMap.set(parentId, []);
    }
    childrenMap.get(parentId)!.push({
      childId,
      edgeType: e.type as 'extends' | 'implements',
    });
  }

  const involvedIds = new Set<string>();
  for (const e of heritageEdges) {
    involvedIds.add(e.source);
    involvedIds.add(e.target);
  }

  const rootIds = Array.from(involvedIds).filter((id) => !hasParent.has(id));

  function buildTreeNode(
    id: string,
    edgeType: 'extends' | 'implements' | 'root',
  ): TreeNode | null {
    const node = nodeMap.get(id);
    if (!node) return null;

    const children: TreeNode[] = [];
    const childEntries = childrenMap.get(id) ?? [];
    for (const entry of childEntries) {
      const child = buildTreeNode(entry.childId, entry.edgeType);
      if (child) children.push(child);
    }

    const isInterface =
      node.label === 'interface' || node.label === 'type_alias';

    return {
      id: node.id,
      name: node.name,
      filePath: node.filePath,
      startLine: node.startLine,
      label: node.label,
      isInterface,
      edgeType,
      children,
    };
  }

  const forest: TreeNode[] = [];
  for (const rootId of rootIds) {
    const tree = buildTreeNode(rootId, 'root');
    if (tree) forest.push(tree);
  }

  forest.sort((a, b) => a.name.localeCompare(b.name));

  return forest;
}
