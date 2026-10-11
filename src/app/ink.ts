import { Point, Stroke } from '../types';

export type PaperStyle = 'lined' | 'grid' | 'dots' | 'blank';
export type PaperTheme = 'warm' | 'white' | 'dark';

export interface View {
  x: number; // screen position of the world origin
  y: number;
  zoom: number;
}

export const MIN_ZOOM = 0.25;
export const MAX_ZOOM = 3;

export const toWorld = (view: View, sx: number, sy: number): Point => ({
  x: (sx - view.x) / view.zoom,
  y: (sy - view.y) / view.zoom,
});

export const toScreen = (view: View, wx: number, wy: number): Point => ({
  x: wx * view.zoom + view.x,
  y: wy * view.zoom + view.y,
});

export function paperColor(theme: PaperTheme): string {
  return theme === 'white' ? '#ffffff' : theme === 'dark' ? '#0f172a' : '#faf9f6';
}

export function drawPaper(ctx: CanvasRenderingContext2D, w: number, h: number, view: View, style: PaperStyle, theme: PaperTheme) {
  const { x: vx, y: vy, zoom } = view;
  ctx.fillStyle = paperColor(theme);
  ctx.fillRect(0, 0, w, h);

  ctx.save();
  ctx.strokeStyle = theme === 'dark' ? 'rgba(255, 255, 255, 0.07)' : 'rgba(15, 23, 42, 0.08)';
  ctx.lineWidth = 1;
  const mod = (a: number, b: number) => ((a % b) + b) % b;

  if (style === 'lined') {
    const spacing = 36 * zoom;
    for (let y = mod(vy, spacing); y < h; y += spacing) {
      ctx.beginPath();
      ctx.moveTo(0, y);
      ctx.lineTo(w, y);
      ctx.stroke();
    }
    const marginX = 80 * zoom + vx;
    if (marginX >= 0 && marginX <= w) {
      ctx.strokeStyle = theme === 'dark' ? 'rgba(244, 63, 94, 0.25)' : 'rgba(244, 63, 94, 0.35)';
      ctx.lineWidth = 1.5;
      ctx.beginPath();
      ctx.moveTo(marginX, 0);
      ctx.lineTo(marginX, h);
      ctx.stroke();
    }
  } else if (style === 'grid') {
    const size = 24 * zoom;
    for (let x = mod(vx, size); x < w; x += size) {
      ctx.beginPath();
      ctx.moveTo(x, 0);
      ctx.lineTo(x, h);
      ctx.stroke();
    }
    for (let y = mod(vy, size); y < h; y += size) {
      ctx.beginPath();
      ctx.moveTo(0, y);
      ctx.lineTo(w, y);
      ctx.stroke();
    }
  } else if (style === 'dots') {
    const spacing = 28 * zoom;
    ctx.fillStyle = theme === 'dark' ? 'rgba(255, 255, 255, 0.15)' : 'rgba(15, 23, 42, 0.18)';
    for (let x = mod(vx, spacing); x < w; x += spacing) {
      for (let y = mod(vy, spacing); y < h; y += spacing) {
        ctx.beginPath();
        ctx.arc(x, y, 1.2 * Math.min(2, Math.max(0.7, zoom)), 0, Math.PI * 2);
        ctx.fill();
      }
    }
  }
  ctx.restore();
}

/** Draw one stroke (world coordinates, the context already has the view transform). */
export function drawStroke(ctx: CanvasRenderingContext2D, stroke: Stroke, highlight = false) {
  const pts = stroke.points;
  if (pts.length < 1) return;
  ctx.save();
  if (stroke.tool === 'highlighter') {
    ctx.globalCompositeOperation = 'multiply';
    ctx.globalAlpha = 0.45;
    ctx.strokeStyle = stroke.color;
    ctx.lineWidth = stroke.width * 5;
    ctx.lineCap = 'square';
    ctx.lineJoin = 'bevel';
  } else {
    ctx.strokeStyle = highlight ? '#4f46e5' : stroke.color;
    ctx.lineWidth = stroke.width;
    ctx.lineCap = 'round';
    ctx.lineJoin = 'round';
  }

  if (pts.length === 1) {
    ctx.beginPath();
    ctx.arc(pts[0].x, pts[0].y, stroke.width / 2, 0, Math.PI * 2);
    ctx.fillStyle = highlight ? '#4f46e5' : stroke.color;
    ctx.fill();
  } else {
    ctx.beginPath();
    ctx.moveTo(pts[0].x, pts[0].y);
    for (let i = 1; i < pts.length; i++) {
      // Quadratic smoothing for a natural ink feel
      const midX = (pts[i - 1].x + pts[i].x) / 2;
      const midY = (pts[i - 1].y + pts[i].y) / 2;
      ctx.quadraticCurveTo(pts[i - 1].x, pts[i - 1].y, midX, midY);
    }
    ctx.lineTo(pts[pts.length - 1].x, pts[pts.length - 1].y);
    ctx.stroke();
  }
  ctx.restore();
}

export function pointInPolygon(pt: Point, polygon: Point[]): boolean {
  let inside = false;
  for (let i = 0, j = polygon.length - 1; i < polygon.length; j = i++) {
    const { x: xi, y: yi } = polygon[i];
    const { x: xj, y: yj } = polygon[j];
    if (yi > pt.y !== yj > pt.y && pt.x < ((xj - xi) * (pt.y - yi)) / (yj - yi) + xi) inside = !inside;
  }
  return inside;
}

export function strokeBounds(strokes: Stroke[]): { x: number; y: number; width: number; height: number } | null {
  let minX = Infinity;
  let minY = Infinity;
  let maxX = -Infinity;
  let maxY = -Infinity;
  for (const s of strokes) {
    for (const p of s.points) {
      minX = Math.min(minX, p.x);
      minY = Math.min(minY, p.y);
      maxX = Math.max(maxX, p.x);
      maxY = Math.max(maxY, p.y);
    }
  }
  if (!Number.isFinite(minX)) return null;
  return { x: minX, y: minY, width: maxX - minX, height: maxY - minY };
}

/** Does the eraser circle at `at` touch this stroke? */
export function strokeTouched(stroke: Stroke, at: Point, radius: number): boolean {
  const r = radius + stroke.width;
  const pts = stroke.points;
  if (pts.length === 1) return Math.hypot(pts[0].x - at.x, pts[0].y - at.y) <= r;
  for (let i = 1; i < pts.length; i++) {
    const a = pts[i - 1];
    const b = pts[i];
    const dx = b.x - a.x;
    const dy = b.y - a.y;
    const len2 = dx * dx + dy * dy;
    const t = len2 > 0 ? Math.max(0, Math.min(1, ((at.x - a.x) * dx + (at.y - a.y) * dy) / len2)) : 0;
    if (Math.hypot(a.x + t * dx - at.x, a.y + t * dy - at.y) <= r) return true;
  }
  return false;
}
