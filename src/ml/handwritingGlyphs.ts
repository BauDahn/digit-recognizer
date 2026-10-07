import { StrokePoint } from './imageOps';

/**
 * A tiny single-stroke "handwriting font" used to write example expressions on the
 * math canvas. Glyphs live in a box 1 unit tall (y grows downwards), and are smoothed
 * with Catmull-Rom splines plus a little random jitter so they look hand-drawn.
 */

type Glyph = { width: number; strokes: Array<Array<[number, number]>> };

const GLYPHS: Record<string, Glyph> = {
  '0': { width: 0.6, strokes: [[[0.3, 0], [0.08, 0.2], [0.03, 0.55], [0.15, 0.9], [0.32, 1], [0.52, 0.85], [0.58, 0.45], [0.48, 0.12], [0.3, 0]]] },
  '1': { width: 0.4, strokes: [[[0.08, 0.2], [0.3, 0], [0.3, 1]]] },
  '2': { width: 0.6, strokes: [[[0.05, 0.22], [0.22, 0.02], [0.45, 0.02], [0.58, 0.2], [0.5, 0.45], [0.02, 1], [0.62, 0.98]]] },
  '3': { width: 0.6, strokes: [[[0.05, 0.1], [0.3, 0], [0.55, 0.12], [0.52, 0.35], [0.25, 0.48], [0.58, 0.62], [0.58, 0.87], [0.3, 1], [0.02, 0.9]]] },
  '4': { width: 0.65, strokes: [[[0.42, 0], [0.02, 0.65], [0.65, 0.65]], [[0.5, 0.3], [0.5, 1]]] },
  '5': { width: 0.6, strokes: [[[0.58, 0], [0.12, 0], [0.06, 0.45], [0.35, 0.38], [0.58, 0.55], [0.57, 0.85], [0.33, 1], [0.02, 0.9]]] },
  '6': { width: 0.6, strokes: [[[0.52, 0], [0.2, 0.3], [0.05, 0.7], [0.25, 1], [0.53, 0.9], [0.55, 0.62], [0.3, 0.5], [0.06, 0.68]]] },
  '7': { width: 0.6, strokes: [[[0.02, 0.02], [0.6, 0], [0.22, 1]]] },
  '8': { width: 0.6, strokes: [[[0.3, 0.5], [0.52, 0.3], [0.48, 0.05], [0.3, 0], [0.1, 0.1], [0.12, 0.32], [0.3, 0.5], [0.57, 0.7], [0.53, 0.95], [0.3, 1], [0.04, 0.9], [0.06, 0.68], [0.3, 0.5]]] },
  '9': { width: 0.6, strokes: [[[0.55, 0.22], [0.38, 0.02], [0.12, 0.06], [0.05, 0.3], [0.25, 0.45], [0.52, 0.32], [0.55, 0.1], [0.53, 0.5], [0.48, 1]]] },
  '+': { width: 0.6, strokes: [[[0, 0.55], [0.6, 0.55]], [[0.3, 0.25], [0.3, 0.85]]] },
  '-': { width: 0.55, strokes: [[[0, 0.55], [0.55, 0.55]]] },
  '=': { width: 0.6, strokes: [[[0, 0.42], [0.6, 0.42]], [[0, 0.68], [0.6, 0.68]]] },
  '/': { width: 0.45, strokes: [[[0.4, 0], [0.05, 1]]] },
  '.': { width: 0.12, strokes: [[[0.04, 0.95], [0.08, 0.99]]] },
  x: { width: 0.55, strokes: [[[0, 0.35], [0.5, 1]], [[0.5, 0.35], [0, 1]]] },
  '(': { width: 0.3, strokes: [[[0.28, -0.05], [0.06, 0.25], [0.02, 0.55], [0.08, 0.85], [0.28, 1.08]]] },
  ')': { width: 0.3, strokes: [[[0.02, -0.05], [0.24, 0.25], [0.28, 0.55], [0.22, 0.85], [0.02, 1.08]]] },
};

function catmullRom(points: StrokePoint[], samplesPerSegment = 6): StrokePoint[] {
  if (points.length < 3) return points;
  const out: StrokePoint[] = [];
  const p = (i: number) => points[Math.max(0, Math.min(points.length - 1, i))];
  for (let i = 0; i < points.length - 1; i++) {
    const [p0, p1, p2, p3] = [p(i - 1), p(i), p(i + 1), p(i + 2)];
    for (let s = 0; s < samplesPerSegment; s++) {
      const t = s / samplesPerSegment;
      const t2 = t * t;
      const t3 = t2 * t;
      const f = (a: number, b: number, c: number, d: number) =>
        0.5 * (2 * b + (-a + c) * t + (2 * a - 5 * b + 4 * c - d) * t2 + (-a + 3 * b - 3 * c + d) * t3);
      out.push({ x: f(p0.x, p1.x, p2.x, p3.x), y: f(p0.y, p1.y, p2.y, p3.y) });
    }
  }
  out.push(points[points.length - 1]);
  return out;
}

/**
 * Write `text` starting at (x, y) (top-left) with glyphs `size` units tall.
 * A '^' makes the next character a raised, smaller exponent.
 */
export function textToStrokes(
  text: string,
  x: number,
  y: number,
  size: number,
  random: () => number = Math.random
): StrokePoint[][] {
  const strokes: StrokePoint[][] = [];
  let cursor = x;
  let superscript = false;
  for (const ch of text) {
    if (ch === '^') {
      superscript = true;
      continue;
    }
    if (ch === ' ') {
      cursor += size * 0.3;
      continue;
    }
    const glyph = GLYPHS[ch];
    if (!glyph) continue;
    const s = superscript ? size * 0.55 : size;
    const top = superscript ? y - size * 0.2 : y;
    const jitter = () => (random() - 0.5) * 0.04;
    for (const stroke of glyph.strokes) {
      const pts = stroke.map(([gx, gy]) => ({ x: cursor + (gx + jitter()) * s, y: top + (gy + jitter()) * s }));
      strokes.push(catmullRom(pts));
    }
    cursor += (glyph.width + 0.3) * s;
    superscript = false;
  }
  return strokes;
}
