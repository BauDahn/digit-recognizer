import { rasterizeStrokes28, StrokePoint } from './imageOps';

/**
 * Generates hand-drawn-looking math operators as vector strokes, then rasterizes them
 * exactly like the math canvas does. MNIST has no operators, so these fill the gap.
 */

export type OperatorSymbol = '+' | '-' | '/' | '=' | '(' | ')' | 'x';

type Rand = () => number;

const between = (r: Rand, a: number, b: number) => a + r() * (b - a);

/** Polyline from a to b with a slight smooth wobble, like a real pen stroke. */
function wobblyLine(r: Rand, ax: number, ay: number, bx: number, by: number, wobble = 0.03): StrokePoint[] {
  const n = 12;
  const phase = r() * Math.PI * 2;
  const amp = between(r, -wobble, wobble);
  const nx = -(by - ay);
  const ny = bx - ax;
  const pts: StrokePoint[] = [];
  for (let i = 0; i <= n; i++) {
    const t = i / n;
    const off = amp * Math.sin(Math.PI * t + phase * 0.2);
    pts.push({ x: ax + (bx - ax) * t + nx * off, y: ay + (by - ay) * t + ny * off });
  }
  return pts;
}

function arc(r: Rand, cx: number, cy: number, rx: number, ry: number, start: number, end: number): StrokePoint[] {
  const n = 16;
  const pts: StrokePoint[] = [];
  for (let i = 0; i <= n; i++) {
    const a = start + ((end - start) * i) / n;
    pts.push({ x: cx + rx * Math.cos(a) + between(r, -0.01, 0.01), y: cy + ry * Math.sin(a) + between(r, -0.01, 0.01) });
  }
  return pts;
}

function rotate(strokes: StrokePoint[][], angle: number): StrokePoint[][] {
  const c = Math.cos(angle);
  const s = Math.sin(angle);
  return strokes.map((st) => st.map((p) => ({ x: p.x * c - p.y * s, y: p.x * s + p.y * c })));
}

export function generateOperatorStrokes(symbol: OperatorSymbol, r: Rand = Math.random): StrokePoint[][] {
  let strokes: StrokePoint[][];
  switch (symbol) {
    case '+': {
      const cx = between(r, -0.1, 0.1);
      const cy = between(r, -0.1, 0.1);
      const w = between(r, 0.35, 0.55);
      const h = between(r, 0.35, 0.55);
      strokes = [wobblyLine(r, -w, cy, w, cy), wobblyLine(r, cx, -h, cx, h)];
      break;
    }
    case '-': {
      const w = between(r, 0.3, 0.6);
      strokes = [wobblyLine(r, -w, 0, w, between(r, -0.05, 0.05), 0.02)];
      break;
    }
    case '=': {
      const w = between(r, 0.35, 0.55);
      const gap = w * between(r, 0.35, 0.8);
      const shift = between(r, -0.08, 0.08);
      strokes = [
        wobblyLine(r, -w, -gap / 2, w, -gap / 2, 0.02),
        wobblyLine(r, -w + shift, gap / 2, w + shift * between(r, 0.5, 1.5), gap / 2, 0.02),
      ];
      break;
    }
    case '/': {
      const w = between(r, 0.15, 0.35);
      strokes = [wobblyLine(r, -w, 0.5, w, -0.5)];
      break;
    }
    case '(': {
      const span = between(r, 0.9, 1.25);
      strokes = [arc(r, 0.35, 0, between(r, 0.25, 0.45), 0.5, Math.PI - span, Math.PI + span)];
      break;
    }
    case ')': {
      const span = between(r, 0.9, 1.25);
      strokes = [arc(r, -0.35, 0, between(r, 0.25, 0.45), 0.5, -span, span)];
      break;
    }
    case 'x': {
      const w = between(r, 0.3, 0.5);
      const h = between(r, 0.3, 0.5);
      // Either two crossing diagonals or two back-to-back curves, like many people write it
      strokes =
        r() < 0.7
          ? [wobblyLine(r, -w, -h, w, h), wobblyLine(r, w, -h, -w, h)]
          : [arc(r, -w, 0, w, h, -Math.PI / 2, Math.PI / 2), arc(r, w, 0, w, h, Math.PI / 2, (3 * Math.PI) / 2)];
      break;
    }
  }
  return rotate(strokes, between(r, -0.15, 0.15));
}

export function generateOperatorImage(symbol: OperatorSymbol, r: Rand = Math.random): Float32Array {
  return rasterizeStrokes28(generateOperatorStrokes(symbol, r), between(r, 3.2, 5.2));
}
