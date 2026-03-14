import { useCallback, useRef, useState, type ReactNode } from 'react';
import { useViewStore } from '@/stores/viewStore';

interface PanelLayoutProps {
  left?: ReactNode;
  center: ReactNode;
  right?: ReactNode;
}

const LEFT_DEFAULT = 260;
const LEFT_MIN = 160;
const LEFT_MAX = 480;

const RIGHT_DEFAULT = 340;
const RIGHT_MIN = 200;
const RIGHT_MAX = 560;

export function PanelLayout({ left, center, right }: PanelLayoutProps) {
  const leftOpen = useViewStore((s) => s.leftSidebarOpen);
  const rightOpen = useViewStore((s) => s.rightPanelOpen);

  const [leftWidth, setLeftWidth] = useState(LEFT_DEFAULT);
  const [rightWidth, setRightWidth] = useState(RIGHT_DEFAULT);

  const dragging = useRef<'left' | 'right' | null>(null);

  const onMouseDown = useCallback(
    (panel: 'left' | 'right') => (e: React.MouseEvent) => {
      e.preventDefault();
      dragging.current = panel;
      const startX = e.clientX;
      const startWidth = panel === 'left' ? leftWidth : rightWidth;

      const onMouseMove = (ev: MouseEvent) => {
        if (!dragging.current) return;
        const delta = ev.clientX - startX;
        if (dragging.current === 'left') {
          setLeftWidth(Math.min(LEFT_MAX, Math.max(LEFT_MIN, startWidth + delta)));
        } else {
          setRightWidth(Math.min(RIGHT_MAX, Math.max(RIGHT_MIN, startWidth - delta)));
        }
      };

      const onMouseUp = () => {
        dragging.current = null;
        document.removeEventListener('mousemove', onMouseMove);
        document.removeEventListener('mouseup', onMouseUp);
        document.body.style.cursor = '';
        document.body.style.userSelect = '';
      };

      document.addEventListener('mousemove', onMouseMove);
      document.addEventListener('mouseup', onMouseUp);
      document.body.style.cursor = 'col-resize';
      document.body.style.userSelect = 'none';
    },
    [leftWidth, rightWidth],
  );

  return (
    <div className="flex h-full w-full overflow-hidden">
      {leftOpen && left && (
        <>
          <div
            className="shrink-0 overflow-y-auto overflow-x-hidden"
            style={{
              width: leftWidth,
              background: 'var(--bg-surface)',
            }}
          >
            {left}
          </div>
          <Divider onMouseDown={onMouseDown('left')} />
        </>
      )}

      <div className="flex-1 min-w-0 overflow-hidden">
        {center}
      </div>

      {rightOpen && right && (
        <>
          <Divider onMouseDown={onMouseDown('right')} />
          <div
            className="shrink-0 overflow-y-auto overflow-x-hidden"
            style={{
              width: rightWidth,
              background: 'var(--bg-surface)',
            }}
          >
            {right}
          </div>
        </>
      )}
    </div>
  );
}

/** 4px hit area with 1px visible border line, centered. */
function Divider({ onMouseDown }: { onMouseDown: (e: React.MouseEvent) => void }) {
  return (
    <div
      onMouseDown={onMouseDown}
      className="shrink-0 relative"
      style={{
        width: 4,
        cursor: 'col-resize',
      }}
    >
      {/* Visible 1px line centered in the 4px hit area */}
      <div
        className="absolute top-0 bottom-0"
        style={{
          left: '50%',
          width: 1,
          transform: 'translateX(-50%)',
          background: 'var(--border)',
        }}
      />
    </div>
  );
}
