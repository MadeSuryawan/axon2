import type { ReactNode } from 'react';

interface EmptyStateAction {
  label: string;
  onClick: () => void;
}

interface EmptyStateProps {
  icon?: ReactNode;
  message: string;
  action?: EmptyStateAction;
}

export function EmptyState({ icon, message, action }: EmptyStateProps) {
  return (
    <div
      style={{
        display: 'flex',
        flexDirection: 'column',
        alignItems: 'center',
        justifyContent: 'center',
        gap: 8,
        width: '100%',
        height: '100%',
        minHeight: 60,
        padding: 16,
        boxSizing: 'border-box',
      }}
    >
      {icon && (
        <div style={{ color: 'var(--text-dimmed)', fontSize: 24, lineHeight: 1 }}>
          {icon}
        </div>
      )}

      <span
        style={{
          color: 'var(--text-dimmed)',
          fontFamily: "'JetBrains Mono', monospace",
          fontSize: 12,
          textAlign: 'center',
        }}
      >
        {message}
      </span>

      {action && (
        <button
          type="button"
          onClick={action.onClick}
          style={{
            background: 'var(--accent-dim)',
            color: 'var(--accent)',
            border: 'none',
            borderRadius: 2,
            padding: '4px 12px',
            fontFamily: "'JetBrains Mono', monospace",
            fontSize: 11,
            fontWeight: 500,
            cursor: 'pointer',
          }}
        >
          {action.label}
        </button>
      )}
    </div>
  );
}
