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
  // Letters and symbols for the examples and tests. Single-stroke shapes written with a steady hand.
  i: { width: 0.25, strokes: [[[0.1, 0.42], [0.12, 1]], [[0.07, 0.1], [0.14, 0.09], [0.15, 0.17], [0.08, 0.18], [0.07, 0.1]]] },
  n: { width: 0.55, strokes: [[[0.05, 0.38], [0.07, 1]], [[0.07, 0.85], [0.1, 0.5], [0.28, 0.35], [0.44, 0.47], [0.47, 1]]] },
  y: { width: 0.55, strokes: [[[0.02, 0.35], [0.27, 1]], [[0.52, 0.35], [0.2, 1.35]]] },
  d: { width: 0.55, strokes: [[[0.5, 0], [0.5, 1]], [[0.5, 0.55], [0.3, 0.38], [0.1, 0.48], [0.04, 0.72], [0.15, 0.95], [0.35, 1], [0.5, 0.88]]] },
  a: { width: 0.5, strokes: [[[0.45, 0.4], [0.22, 0.33], [0.05, 0.55], [0.1, 0.9], [0.3, 1], [0.45, 0.85]], [[0.45, 0.35], [0.45, 1]]] },
  t: { width: 0.4, strokes: [[[0.2, 0.05], [0.2, 0.95], [0.35, 1]], [[0.02, 0.38], [0.4, 0.38]]] },
  'π': { width: 0.65, strokes: [[[0.0, 0.4], [0.65, 0.4]], [[0.2, 0.4], [0.18, 1]], [[0.45, 0.4], [0.5, 1]]] },
  '∞': { width: 0.85, strokes: [[[0.42, 0.62], [0.65, 0.4], [0.82, 0.55], [0.65, 0.85], [0.42, 0.62], [0.2, 0.4], [0.03, 0.55], [0.2, 0.85], [0.42, 0.62]]] },
  'Σ': { width: 0.6, strokes: [[[0.58, 0.05], [0.04, 0.05], [0.32, 0.5], [0.04, 0.95], [0.6, 0.95]]] },
  '∏': { width: 0.7, strokes: [[[0, 0.08], [0.7, 0.08]], [[0.12, 0.08], [0.1, 0.98]], [[0.58, 0.08], [0.6, 0.98]]] },
  '∫': { width: 0.35, strokes: [[[0.38, 0.05], [0.3, 0], [0.22, 0.1], [0.2, 0.5], [0.17, 0.92], [0.08, 1.02], [0, 0.95]]] },
  '∂': { width: 0.55, strokes: [[[0.47, 0.55], [0.25, 0.5], [0.06, 0.68], [0.14, 0.95], [0.36, 1], [0.53, 0.8], [0.56, 0.45], [0.48, 0.15], [0.3, 0], [0.12, 0.1]]] },
  '<': { width: 0.5, strokes: [[[0.45, 0.25], [0.05, 0.55], [0.45, 0.85]]] },
  '>': { width: 0.5, strokes: [[[0.05, 0.25], [0.45, 0.55], [0.05, 0.85]]] },
  '*': { width: 0.12, strokes: [[[0.04, 0.55], [0.08, 0.59]]] },
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

/** A piece of handwriting: strokes relative to its top-left corner and where its baseline is. */
interface Piece {
  strokes: StrokePoint[][];
  width: number;
  height: number;
  /** distance from the top to the baseline */
  baseline: number;
}

type Rand = () => number;

function glyphPiece(ch: string, size: number, random: Rand): Piece | null {
  const glyph = GLYPHS[ch];
  if (!glyph) return null;
  const jitter = () => (random() - 0.5) * 0.04;
  const strokes = glyph.strokes.map((stroke) =>
    catmullRom(stroke.map(([gx, gy]) => ({ x: (gx + jitter()) * size, y: (gy + jitter()) * size })))
  );
  return { strokes, width: glyph.width * size, height: size, baseline: size };
}

const shift = (strokes: StrokePoint[][], dx: number, dy: number) => strokes.map((s) => s.map((p) => ({ x: p.x + dx, y: p.y + dy })));

/** Put pieces side by side on a common baseline. */
function sequence(pieces: Piece[], gap: number): Piece {
  const ascent = Math.max(0, ...pieces.map((p) => p.baseline));
  const descent = Math.max(0, ...pieces.map((p) => p.height - p.baseline));
  const strokes: StrokePoint[][] = [];
  let cursor = 0;
  for (const p of pieces) {
    strokes.push(...shift(p.strokes, cursor, ascent - p.baseline));
    cursor += p.width + gap;
  }
  return { strokes, width: Math.max(0, cursor - gap), height: ascent + descent, baseline: ascent };
}

/** Pieces stacked vertically and centered, the baseline of the one at `axisIndex` is the result's. */
function stack(pieces: Piece[], gaps: number[], axis: (parts: number[]) => number): Piece {
  const width = Math.max(...pieces.map((p) => p.width));
  const strokes: StrokePoint[][] = [];
  const tops: number[] = [];
  let y = 0;
  pieces.forEach((p, i) => {
    tops.push(y);
    strokes.push(...shift(p.strokes, (width - p.width) / 2, y));
    y += p.height + (gaps[i] ?? 0);
  });
  return { strokes, width, height: y - (gaps[pieces.length - 1] ?? 0), baseline: axis(tops) };
}

/**
 * Handwriting for a small LaTeX-like language, used for the examples on the canvas and in tests:
 *   digits, letters, + - = / ( ) . < > π ∞ ∫ ∂ and  x^2  x_i  x^{10}  \\frac{a}{b}  \\sqrt{a}
 *   \\sum_{i=1}^{n}  \\prod_{k=1}^{3}  \\int_{0}^{2}  \\partial  \\pi
 */
class Writer {
  private pos = 0;
  constructor(
    private src: string,
    private random: Rand
  ) {}

  private readGroup(): string {
    if (this.src[this.pos] !== '{') return this.src[this.pos++] ?? '';
    let depth = 0;
    const start = ++this.pos;
    for (; this.pos < this.src.length; this.pos++) {
      if (this.src[this.pos] === '{') depth++;
      if (this.src[this.pos] === '}') {
        if (depth === 0) break;
        depth--;
      }
    }
    return this.src.slice(start, this.pos++);
  }

  private sub(text: string, size: number): Piece {
    return new Writer(text, this.random).write(size);
  }

  write(size: number): Piece {
    const pieces: Piece[] = [];
    while (this.pos < this.src.length) {
      const ch = this.src[this.pos++];
      if (ch === ' ') {
        pieces.push({ strokes: [], width: size * 0.3, height: size, baseline: size });
        continue;
      }
      let piece: Piece | null;
      if (ch === '\\') {
        const name = /^[a-z]+/.exec(this.src.slice(this.pos))?.[0] ?? '';
        this.pos += name.length;
        piece = this.command(name, size);
      } else {
        piece = glyphPiece(ch, size, this.random);
      }
      if (!piece) continue;

      // exponents and indices (x^2_i, x_i^2)
      let sup: Piece | undefined;
      let sub: Piece | undefined;
      while (this.src[this.pos] === '^' || this.src[this.pos] === '_') {
        const kind = this.src[this.pos++];
        const arg = this.sub(this.readGroup(), size * 0.55);
        if (kind === '^') sup = arg;
        else sub = arg;
      }
      if (sup || sub) piece = this.withScripts(piece, sup, sub, size);
      pieces.push(piece);
    }
    return sequence(pieces, size * 0.14);
  }

  private withScripts(base: Piece, sup: Piece | undefined, sub: Piece | undefined, size: number): Piece {
    const strokes = [...base.strokes];
    let width = base.width;
    let top = 0;
    let bottom = base.height;
    const x = base.width + size * 0.05;
    if (sup) {
      const y = base.baseline - base.height * 0.95 - sup.height * 0.15;
      strokes.push(...shift(sup.strokes, x, y));
      width = Math.max(width, x + sup.width);
      top = Math.min(top, y);
    }
    if (sub) {
      const y = base.baseline - sub.baseline + size * 0.3;
      strokes.push(...shift(sub.strokes, x, y));
      width = Math.max(width, x + sub.width);
      bottom = Math.max(bottom, y + sub.height);
    }
    return { strokes: shift(strokes, 0, -top), width, height: bottom - top, baseline: base.baseline - top };
  }

  private command(name: string, size: number): Piece | null {
    switch (name) {
      case 'pi':
        return glyphPiece('π', size, this.random);
      case 'partial':
        return glyphPiece('∂', size, this.random);
      case 'infty':
        return glyphPiece('∞', size, this.random);
      case 'times':
        return glyphPiece('x', size, this.random);
      case 'frac': {
        const num = this.sub(this.readGroup(), size * 0.8);
        const den = this.sub(this.readGroup(), size * 0.8);
        const width = Math.max(num.width, den.width) + size * 0.3;
        const gap = size * 0.15;
        const barY = num.height + gap;
        const strokes: StrokePoint[][] = [
          ...shift(num.strokes, (width - num.width) / 2, 0),
          [
            { x: 0, y: barY },
            { x: width, y: barY },
          ],
          ...shift(den.strokes, (width - den.width) / 2, barY + gap),
        ];
        // the bar sits half a line above the baseline, like the minus sign of a sum
        return { strokes, width, height: barY + gap + den.height, baseline: barY + size * 0.5 };
      }
      case 'sqrt': {
        const body = this.sub(this.readGroup(), size);
        const h = body.height * 1.1;
        const lead = size * 0.5;
        const strokes: StrokePoint[][] = [
          catmullRom(
            [
              { x: 0, y: h * 0.55 },
              { x: lead * 0.25, y: h * 0.5 },
              { x: lead * 0.55, y: h },
              { x: lead, y: 0 },
              { x: lead + body.width + size * 0.15, y: 0 },
            ],
            5
          ),
          ...shift(body.strokes, lead + size * 0.05, h - body.height),
        ];
        return { strokes, width: lead + body.width + size * 0.2, height: h, baseline: h - (body.height - body.baseline) };
      }
      case 'sum':
      case 'prod':
      case 'int': {
        const big = name === 'int' ? size * 1.6 : size * 1.25;
        const glyph = glyphPiece(name === 'sum' ? 'Σ' : name === 'prod' ? '∏' : '∫', big, this.random)!;
        let lower: Piece | undefined;
        let upper: Piece | undefined;
        while (this.src[this.pos] === '^' || this.src[this.pos] === '_') {
          const kind = this.src[this.pos++];
          const arg = this.sub(this.readGroup(), size * 0.55);
          if (kind === '^') upper = arg;
          else lower = arg;
        }
        if (name === 'int') {
          // limits to the right of ∫: upper at the top, lower at the bottom
          const strokes = [...glyph.strokes];
          let width = glyph.width;
          if (upper) {
            strokes.push(...shift(upper.strokes, glyph.width + size * 0.08, -size * 0.1));
            width = Math.max(width, glyph.width + size * 0.08 + upper.width);
          }
          if (lower) {
            strokes.push(...shift(lower.strokes, glyph.width - size * 0.12, glyph.height - lower.height * 0.6));
            width = Math.max(width, glyph.width - size * 0.12 + lower.width);
          }
          return { strokes, width, height: glyph.height + size * 0.1, baseline: glyph.baseline - size * 0.15 };
        }
        const parts = [upper, glyph, lower].filter((p): p is Piece => !!p);
        return stack(parts, parts.map((_, i) => (i < parts.length - 1 ? size * 0.12 : 0)), (tops) => tops[parts.indexOf(glyph)] + glyph.height * 0.75);
      }
      default:
        return null;
    }
  }
}

/**
 * Write `text` starting at (x, y) (top-left) with glyphs `size` units tall. Besides plain
 * characters it understands the LaTeX-like commands described at the top of `Writer`.
 */
export function textToStrokes(text: string, x: number, y: number, size: number, random: Rand = Math.random): StrokePoint[][] {
  return shift(new Writer(text, random).write(size).strokes, x, y);
}
