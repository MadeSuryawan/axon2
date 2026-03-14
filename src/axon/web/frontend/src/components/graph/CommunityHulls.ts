import type { AbstractGraph, Attributes } from 'graphology-types';
import type { Community } from '@/types';

export interface Point {
  x: number;
  y: number;
}

export interface HullData {
  communityId: string;
  name: string;
  color: string;
  /** Ordered vertices of the convex hull polygon. */
  points: Point[];
  /** Geometric centroid of the hull. */
  centroid: Point;
}

const HULL_COLORS = [
  '#61AFEF', '#E5C07B', '#C678DD', '#98C379', '#56B6C2', '#E06C75',
  '#BE85C8', '#5EA8A0', '#D19A66', '#7EC4FF', '#E98B93', '#B0D494',
];

export function computeHulls(
  graph: AbstractGraph<Attributes, Attributes, Attributes>,
  communities: Community[],
): HullData[] {
  const hulls: HullData[] = [];

  for (let i = 0; i < communities.length; i++) {
    const community = communities[i];
    const points: Point[] = [];

    for (const memberId of community.members) {
      if (!graph.hasNode(memberId)) continue;

      const attrs = graph.getNodeAttributes(memberId);
      const x = attrs.x as number | undefined;
      const y = attrs.y as number | undefined;

      if (x !== undefined && y !== undefined && isFinite(x) && isFinite(y)) {
        points.push({ x, y });
      }
    }

    // Need at least 3 points to form a meaningful hull.
    if (points.length < 3) continue;

    const hull = convexHull(points);
    if (hull.length < 3) continue;

    const centroid = computeCentroid(hull);

    hulls.push({
      communityId: community.id,
      name: community.name,
      color: HULL_COLORS[i % HULL_COLORS.length],
      points: hull,
      centroid,
    });
  }

  return hulls;
}

// Graham scan convex hull
function convexHull(points: Point[]): Point[] {
  if (points.length < 3) return [...points];

  // Find the lowest point (and leftmost if tied).
  let pivot = points[0];
  for (let i = 1; i < points.length; i++) {
    const p = points[i];
    if (p.y < pivot.y || (p.y === pivot.y && p.x < pivot.x)) {
      pivot = p;
    }
  }

  // Sort points by polar angle with respect to the pivot.
  const sorted = points
    .filter((p) => p !== pivot)
    .sort((a, b) => {
      const angleA = Math.atan2(a.y - pivot.y, a.x - pivot.x);
      const angleB = Math.atan2(b.y - pivot.y, b.x - pivot.x);
      if (angleA !== angleB) return angleA - angleB;
      // If same angle, closer point first.
      const distA = (a.x - pivot.x) ** 2 + (a.y - pivot.y) ** 2;
      const distB = (b.x - pivot.x) ** 2 + (b.y - pivot.y) ** 2;
      return distA - distB;
    });

  const stack: Point[] = [pivot];

  for (const point of sorted) {
    // Remove points that make a clockwise turn.
    while (stack.length >= 2) {
      const top = stack[stack.length - 1];
      const belowTop = stack[stack.length - 2];
      if (cross(belowTop, top, point) <= 0) {
        stack.pop();
      } else {
        break;
      }
    }
    stack.push(point);
  }

  return stack;
}

/** Cross product of vectors (O->A) and (O->B). Positive = counter-clockwise. */
function cross(o: Point, a: Point, b: Point): number {
  return (a.x - o.x) * (b.y - o.y) - (a.y - o.y) * (b.x - o.x);
}

function computeCentroid(points: Point[]): Point {
  let cx = 0;
  let cy = 0;
  for (const p of points) {
    cx += p.x;
    cy += p.y;
  }
  return {
    x: cx / points.length,
    y: cy / points.length,
  };
}
