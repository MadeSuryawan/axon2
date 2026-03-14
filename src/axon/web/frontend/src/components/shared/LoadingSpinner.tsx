interface LoadingSpinnerProps {
  message?: string;
}

export function LoadingSpinner({ message = 'Loading...' }: LoadingSpinnerProps) {
  return (
    <div
      style={{
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        gap: 8,
        width: '100%',
        height: '100%',
        minHeight: 40,
      }}
    >
      <span
        style={{
          display: 'inline-block',
          width: 8,
          height: 8,
          borderRadius: '50%',
          background: 'var(--accent)',
          animation: 'axon-pulse 1.4s ease-in-out infinite',
          flexShrink: 0,
        }}
      />
      <span
        style={{
          color: 'var(--text-secondary)',
          fontFamily: "'JetBrains Mono', monospace",
          fontSize: 11,
        }}
      >
        {message}
      </span>

      <style>{`
        @keyframes axon-pulse {
          0%, 100% { transform: scale(0.8); opacity: 0.5; }
          50%      { transform: scale(1.2); opacity: 1; }
        }
      `}</style>
    </div>
  );
}
