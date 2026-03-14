import { Component } from 'react';
import type { ErrorInfo, ReactNode } from 'react';

interface ErrorBoundaryProps {
  children: ReactNode;
}

interface ErrorBoundaryState {
  hasError: boolean;
  error: Error | null;
}

export class ErrorBoundary extends Component<ErrorBoundaryProps, ErrorBoundaryState> {
  constructor(props: ErrorBoundaryProps) {
    super(props);
    this.state = { hasError: false, error: null };
  }

  static getDerivedStateFromError(error: Error): ErrorBoundaryState {
    return { hasError: true, error };
  }

  componentDidCatch(error: Error, info: ErrorInfo): void {
    console.error('[ErrorBoundary]', error, info.componentStack);
  }

  private handleRetry = (): void => {
    this.setState({ hasError: false, error: null });
  };

  render() {
    if (this.state.hasError) {
      return (
        <div
          style={{
            display: 'flex',
            flexDirection: 'column',
            alignItems: 'center',
            justifyContent: 'center',
            gap: 12,
            width: '100%',
            height: '100%',
            padding: 24,
            boxSizing: 'border-box',
          }}
        >
          <span
            style={{
              color: 'var(--danger)',
              fontFamily: "'JetBrains Mono', monospace",
              fontSize: 13,
              fontWeight: 600,
            }}
          >
            Something went wrong
          </span>

          {this.state.error && (
            <pre
              style={{
                color: 'var(--danger)',
                fontFamily: "'JetBrains Mono', monospace",
                fontSize: 11,
                maxWidth: '100%',
                overflow: 'auto',
                whiteSpace: 'pre-wrap',
                wordBreak: 'break-word',
                background: 'var(--bg-surface)',
                border: '1px solid var(--danger)',
                borderRadius: 2,
                padding: 12,
                margin: 0,
              }}
            >
              {this.state.error.message}
            </pre>
          )}

          <button
            type="button"
            onClick={this.handleRetry}
            style={{
              background: 'var(--accent-dim)',
              color: 'var(--accent)',
              border: 'none',
              borderRadius: 2,
              padding: '6px 16px',
              fontFamily: "'JetBrains Mono', monospace",
              fontSize: 11,
              fontWeight: 500,
              cursor: 'pointer',
            }}
          >
            Retry
          </button>
        </div>
      );
    }

    return this.props.children;
  }
}
