import { useEffect, useRef, useCallback } from 'react';
import type Sigma from 'sigma';

interface MinimapProps {
  sigma: Sigma;
}

const MINIMAP_W = 160;
const MINIMAP_H = 120;
const PAD = 6;

function getCSSVar(name: string, fallback: string): string {
  if (typeof document === 'undefined') return fallback;
  const value = getComputedStyle(document.documentElement).getPropertyValue(name).trim();
  return value || fallback;
}

export function Minimap({ sigma }: MinimapProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const draggingRef = useRef(false);

  const getGraphBounds = useCallback(() => {
    const graph = sigma.getGraph();
    let minX = Infinity;
    let maxX = -Infinity;
    let minY = Infinity;
    let maxY = -Infinity;

    graph.forEachNode((_id, attrs) => {
      const x = attrs.x as number;
      const y = attrs.y as number;
      if (x < minX) minX = x;
      if (x > maxX) maxX = x;
      if (y < minY) minY = y;
      if (y > maxY) maxY = y;
    });

    if (!isFinite(minX) || minX === maxX) {
      minX = -1;
      maxX = 1;
    }
    if (!isFinite(minY) || minY === maxY) {
      minY = -1;
      maxY = 1;
    }

    return { minX, maxX, minY, maxY };
  }, [sigma]);

  const graphToMinimap = useCallback(
    (
      gx: number,
      gy: number,
      bounds: { minX: number; maxX: number; minY: number; maxY: number },
    ): { mx: number; my: number } => {
      const rangeX = bounds.maxX - bounds.minX;
      const rangeY = bounds.maxY - bounds.minY;
      const drawW = MINIMAP_W - PAD * 2;
      const drawH = MINIMAP_H - PAD * 2;
      const mx = PAD + ((gx - bounds.minX) / rangeX) * drawW;
      const my = PAD + ((gy - bounds.minY) / rangeY) * drawH;
      return { mx, my };
    },
    [],
  );

  const minimapToGraph = useCallback(
    (
      mx: number,
      my: number,
      bounds: { minX: number; maxX: number; minY: number; maxY: number },
    ): { gx: number; gy: number } => {
      const rangeX = bounds.maxX - bounds.minX;
      const rangeY = bounds.maxY - bounds.minY;
      const drawW = MINIMAP_W - PAD * 2;
      const drawH = MINIMAP_H - PAD * 2;
      const gx = bounds.minX + ((mx - PAD) / drawW) * rangeX;
      const gy = bounds.minY + ((my - PAD) / drawH) * rangeY;
      return { gx, gy };
    },
    [],
  );

  const paint = useCallback(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const graph = sigma.getGraph();
    const bounds = getGraphBounds();

    const bgColor = getCSSVar('--bg-surface', '#0d1117');
    const borderColor = getCSSVar('--border', '#1e2a3a');
    const accentColor = getCSSVar('--accent', '#39d353');

    ctx.clearRect(0, 0, MINIMAP_W, MINIMAP_H);
    ctx.fillStyle = bgColor;
    ctx.fillRect(0, 0, MINIMAP_W, MINIMAP_H);

    ctx.strokeStyle = borderColor;
    ctx.lineWidth = 1;
    ctx.strokeRect(0.5, 0.5, MINIMAP_W - 1, MINIMAP_H - 1);

    graph.forEachNode((id) => {
      const display = sigma.getNodeDisplayData(id);
      if (!display || display.hidden) return;

      const { mx, my } = graphToMinimap(display.x, display.y, bounds);
      ctx.fillStyle = display.color;
      ctx.fillRect(Math.round(mx) - 1, Math.round(my) - 1, 2, 2);
    });

    // Convert viewport corners from viewport pixels to graph space, then to
    // minimap coordinates to draw the visible-area rectangle.
    const dims = sigma.getDimensions();
    const topLeft = sigma.viewportToGraph({ x: 0, y: 0 });
    const bottomRight = sigma.viewportToGraph({ x: dims.width, y: dims.height });

    const tl = graphToMinimap(topLeft.x, topLeft.y, bounds);
    const br = graphToMinimap(bottomRight.x, bottomRight.y, bounds);

    const rectX = Math.min(tl.mx, br.mx);
    const rectY = Math.min(tl.my, br.my);
    const rectW = Math.abs(br.mx - tl.mx);
    const rectH = Math.abs(br.my - tl.my);

    ctx.strokeStyle = accentColor;
    ctx.lineWidth = 1.5;
    ctx.strokeRect(rectX, rectY, rectW, rectH);
  }, [sigma, getGraphBounds, graphToMinimap]);

  const navigateTo = useCallback(
    (clientX: number, clientY: number) => {
      const canvas = canvasRef.current;
      if (!canvas) return;

      const rect = canvas.getBoundingClientRect();
      const mx = clientX - rect.left;
      const my = clientY - rect.top;
      const bounds = getGraphBounds();
      const { gx, gy } = minimapToGraph(mx, my, bounds);

      const camera = sigma.getCamera();
      camera.setState({ x: 0.5, y: 0.5 }); // reset to center first

      // Convert the target graph coords to the normalized sigma camera space.
      // Sigma camera x/y is in [0,1] "framed graph" space after normalization.
      const viewportCoords = sigma.graphToViewport({ x: gx, y: gy });
      const dims = sigma.getDimensions();

      // The camera state x,y corresponds to the center of the viewport in
      // framed graph space. We shift the camera so that (gx,gy) lands at center.
      const currentState = camera.getState();
      const centerViewport = { x: dims.width / 2, y: dims.height / 2 };

      const dvx = viewportCoords.x - centerViewport.x;
      const dvy = viewportCoords.y - centerViewport.y;

      const ratio = sigma.getGraphToViewportRatio();
      if (ratio === 0) return;

      camera.animate(
        {
          x: currentState.x + dvx / ratio / dims.width,
          y: currentState.y + dvy / ratio / dims.height,
        },
        { duration: 150 },
      );
    },
    [sigma, getGraphBounds, minimapToGraph],
  );

  useEffect(() => {
    const onRender = () => paint();
    sigma.on('afterRender', onRender);
    paint();
    return () => {
      sigma.off('afterRender', onRender);
    };
  }, [sigma, paint]);

  const handleMouseDown = useCallback(
    (e: React.MouseEvent<HTMLCanvasElement>) => {
      e.preventDefault();
      e.stopPropagation();
      draggingRef.current = true;
      navigateTo(e.clientX, e.clientY);
    },
    [navigateTo],
  );

  const handleMouseMove = useCallback(
    (e: React.MouseEvent<HTMLCanvasElement>) => {
      if (!draggingRef.current) return;
      e.preventDefault();
      e.stopPropagation();
      navigateTo(e.clientX, e.clientY);
    },
    [navigateTo],
  );

  const handleMouseUp = useCallback((e: React.MouseEvent<HTMLCanvasElement>) => {
    e.preventDefault();
    e.stopPropagation();
    draggingRef.current = false;
  }, []);

  const handleMouseLeave = useCallback(() => {
    draggingRef.current = false;
  }, []);

  return (
    <canvas
      ref={canvasRef}
      width={MINIMAP_W}
      height={MINIMAP_H}
      onMouseDown={handleMouseDown}
      onMouseMove={handleMouseMove}
      onMouseUp={handleMouseUp}
      onMouseLeave={handleMouseLeave}
      style={{
        position: 'absolute',
        bottom: 8,
        right: 8,
        width: MINIMAP_W,
        height: MINIMAP_H,
        zIndex: 10,
        cursor: 'crosshair',
        borderRadius: 'var(--radius, 2px)',
      }}
    />
  );
}
