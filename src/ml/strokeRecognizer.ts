import { Box, rasterizeStrokes28, StrokePoint } from './imageOps';
import { layoutSymbols, LayoutNode, RecognizedSymbol } from './mathLayout';
import { LoadedModel } from './modelFile';

/**
 * Turns handwritten strokes from the math canvas into math:
 *
 * 1. Group strokes into symbols. Strokes that touch or cross belong together ('+', 'x', 'π'),
 *    plus the few multi-stroke symbols made of separate parts: '=' (two bars), the dot of 'i' or '÷'.
 * 2. Tiny marks are dots: '.' on the baseline, '·' (multiplication) in the middle.
 * 3. Every other symbol is rasterized to 28x28 like an MNIST image and classified by the network.
 * 4. mathLayout.ts reads the page layout (fractions, roots, limits, exponents, indices).
 */

export type { RecognizedSymbol } from './mathLayout';

export interface StrokeRecognitionResult {
  symbols: RecognizedSymbol[];
  /** Text for the math parser, e.g. "\frac(1,2)+x^(2)" */
  expression: string;
  latex: string;
  layout: LayoutNode;
}

interface Group {
  strokes: number[];
  minX: number;
  maxX: number;
  minY: number;
  maxY: number;
}

interface StrokeInfo {
  points: StrokePoint[]; // resampled
  minX: number;
  maxX: number;
  minY: number;
  maxY: number;
  w: number;
  h: number;
  size: number;
}

function strokeBox(points: StrokePoint[]) {
  let minX = Infinity;
  let maxX = -Infinity;
  let minY = Infinity;
  let maxY = -Infinity;
  for (const p of points) {
    minX = Math.min(minX, p.x);
    maxX = Math.max(maxX, p.x);
    minY = Math.min(minY, p.y);
    maxY = Math.max(maxY, p.y);
  }
  return { minX, maxX, minY, maxY };
}

/** At most `n` points evenly spread along the stroke, enough to measure distances between strokes. */
function resample(points: StrokePoint[], n = 28): StrokePoint[] {
  if (points.length <= n) return points;
  const out: StrokePoint[] = [];
  for (let i = 0; i < n; i++) out.push(points[Math.round((i * (points.length - 1)) / (n - 1))]);
  return out;
}

function pointSegmentDistance(p: StrokePoint, a: StrokePoint, b: StrokePoint): number {
  const dx = b.x - a.x;
  const dy = b.y - a.y;
  const len2 = dx * dx + dy * dy;
  const t = len2 > 0 ? Math.max(0, Math.min(1, ((p.x - a.x) * dx + (p.y - a.y) * dy) / len2)) : 0;
  return Math.hypot(a.x + t * dx - p.x, a.y + t * dy - p.y);
}

function segmentsCross(a: StrokePoint, b: StrokePoint, c: StrokePoint, d: StrokePoint): boolean {
  const o = (p: StrokePoint, q: StrokePoint, r: StrokePoint) => Math.sign((q.x - p.x) * (r.y - p.y) - (q.y - p.y) * (r.x - p.x));
  return o(a, b, c) !== o(a, b, d) && o(c, d, a) !== o(c, d, b);
}

/** Smallest distance between two strokes (0 if they cross). */
function strokeDistance(a: StrokeInfo, b: StrokeInfo): number {
  let best = Infinity;
  for (let i = 0; i < a.points.length; i++) {
    const a0 = a.points[i];
    const a1 = a.points[Math.min(i + 1, a.points.length - 1)];
    for (let j = 0; j < b.points.length; j++) {
      const b0 = b.points[j];
      const b1 = b.points[Math.min(j + 1, b.points.length - 1)];
      if (a0 !== a1 && b0 !== b1 && segmentsCross(a0, a1, b0, b1)) return 0;
      best = Math.min(best, pointSegmentDistance(a0, b0, b1), pointSegmentDistance(b0, a0, a1));
    }
  }
  return best;
}

const isFlat = (s: { w: number; h: number }) => s.h <= 0.35 * s.w;

/** Do two strokes belong to the same symbol? `ref` is a typical symbol height on the page. */
function sameSymbol(a: StrokeInfo, b: StrokeInfo, ref: number): boolean {
  const tinyA = a.size < 0.22 * ref;
  const tinyB = b.size < 0.22 * ref;

  // The dot of i, j, !, ÷ : a tiny mark right above or below the middle of a bigger stroke
  if (tinyA !== tinyB) {
    const [dot, stem] = tinyA ? [a, b] : [b, a];
    const dx = Math.abs((dot.minX + dot.maxX) / 2 - (stem.minX + stem.maxX) / 2);
    const gap = Math.max(stem.minY - dot.maxY, dot.minY - stem.maxY, 0);
    const insideX = dx < 0.35 * Math.max(stem.w, 0.3 * stem.size);
    // not between the lines of an '=' ... only above/below, never beside
    return insideX && gap < 0.6 * stem.size && (dot.maxY < stem.minY + 0.2 * stem.h || dot.minY > stem.maxY - 0.2 * stem.h);
  }
  if (tinyA && tinyB) return false;

  // Two flat bars, one over the other: '=' however wide the gap between them
  if (isFlat(a) && isFlat(b)) {
    const overlap = Math.min(a.maxX, b.maxX) - Math.max(a.minX, b.minX);
    const ratio = a.w / b.w;
    const gap = Math.max(a.minY - b.maxY, b.minY - a.maxY, 0);
    if (overlap >= 0.6 * Math.min(a.w, b.w) && ratio > 0.55 && ratio < 1.8 && gap < 0.9 * Math.min(a.w, b.w)) return true;
  }

  // Parts of one symbol touch or cross each other. A stroke much bigger than the other is a different
  // thing when it only comes close (a root's overline and the number under it, a fraction bar and
  // the numerator written right above it).
  const ratio = Math.max(a.size, b.size) / Math.max(1, Math.min(a.size, b.size));
  const smaller = Math.min(a.size, b.size);
  const d = strokeDistance(a, b);
  if (d <= 0.03 * smaller) return ratio <= 3.2;
  return ratio <= 1.5 && d < 0.08 * smaller;
}

export function groupStrokes(strokes: StrokePoint[][]): Group[] {
  const infos: Array<StrokeInfo | null> = strokes.map((raw) => {
    const box = strokeBox(raw);
    if (!Number.isFinite(box.minX)) return null;
    const w = box.maxX - box.minX;
    const h = box.maxY - box.minY;
    return { ...box, points: resample(raw), w, h, size: Math.max(w, h) };
  });
  const valid = infos.map((info, i) => (info ? i : -1)).filter((i) => i >= 0);
  if (valid.length === 0) return [];

  // A typical symbol height: the median of the larger half of the strokes
  const sizes = valid.map((i) => infos[i]!.size).sort((a, b) => a - b);
  const ref = sizes[Math.floor(sizes.length * 0.65)] || sizes[sizes.length - 1];

  const parent = new Map(valid.map((i) => [i, i]));
  const find = (i: number): number => (parent.get(i) === i ? i : (parent.set(i, find(parent.get(i)!)), parent.get(i)!));
  for (let x = 0; x < valid.length; x++) {
    for (let y = x + 1; y < valid.length; y++) {
      const a = valid[x];
      const b = valid[y];
      if (find(a) !== find(b) && sameSymbol(infos[a]!, infos[b]!, ref)) parent.set(find(a), find(b));
    }
  }

  const groups = new Map<number, Group>();
  for (const i of valid) {
    const root = find(i);
    const info = infos[i]!;
    const g = groups.get(root);
    if (!g) {
      groups.set(root, { strokes: [i], minX: info.minX, maxX: info.maxX, minY: info.minY, maxY: info.maxY });
    } else {
      g.strokes.push(i);
      g.minX = Math.min(g.minX, info.minX);
      g.maxX = Math.max(g.maxX, info.maxX);
      g.minY = Math.min(g.minY, info.minY);
      g.maxY = Math.max(g.maxY, info.maxY);
    }
  }
  return [...groups.values()].sort((a, b) => a.minX - b.minX);
}

function median(values: number[]): number {
  if (values.length === 0) return 0;
  const sorted = [...values].sort((a, b) => a - b);
  return sorted[Math.floor(sorted.length / 2)];
}

function topCandidates(probabilities: Float32Array, classes: string[], allowed?: Set<string>, n = 5) {
  return classes
    .map((char, i) => ({ char, confidence: probabilities[i] }))
    .filter((c) => !allowed || allowed.has(c.char))
    .sort((a, b) => b.confidence - a.confidence)
    .slice(0, n);
}

/** A very flat shape: the 28x28 image loses its size, so only these marks are possible. */
const FLAT_SYMBOLS = new Set(['-', '→', '←', '↔', '⇒', '⇐', '⇔', '↦', '∼', '≈']);
const EQUALS = new Set(['=']);
const TRIPLE_BAR = new Set(['≡', '=']);

/** How many flat bars, one over the other without touching: 2 is '=', 3 is '≡'. */
function stackedBars(g: Group, strokes: StrokePoint[][]): number {
  if (g.strokes.length < 2 || g.strokes.length > 3) return 0;
  const boxes = g.strokes.map((i) => strokeBox(strokes[i])).sort((p, q) => p.minY - q.minY);
  const flat = (s: (typeof boxes)[number]) => s.maxY - s.minY < 0.35 * Math.max(1, s.maxX - s.minX);
  for (let i = 0; i < boxes.length; i++) {
    if (!flat(boxes[i]) || (i > 0 && boxes[i - 1].maxY >= boxes[i].minY)) return 0;
  }
  return boxes.length;
}

export function recognizeStrokes(strokes: StrokePoint[][], model: LoadedModel): StrokeRecognitionResult {
  const groups = groupStrokes(strokes);
  const empty: LayoutNode = { type: 'row', items: [] };
  if (groups.length === 0) return { symbols: [], expression: '', latex: '', layout: empty };

  // Typical symbol height, ignoring flat symbols and dots
  const heights = groups.map((g) => g.maxY - g.minY);
  const tallest = Math.max(...heights, ...groups.map((g) => g.maxX - g.minX));
  const lineHeight = Math.max(1, median(heights.filter((h) => h > 0.4 * Math.max(...heights))), 0.2 * tallest);
  const lineTop = median(groups.map((g) => g.minY));
  const lineBottom = lineTop + lineHeight;

  const symbols: RecognizedSymbol[] = [];
  for (const g of groups) {
    const w = g.maxX - g.minX;
    const h = g.maxY - g.minY;
    const box: Box = { x: g.minX, y: g.minY, width: w, height: h };
    const base = { id: symbols.length, box, strokeIndices: g.strokes, role: 'base' as const };

    if (Math.max(w, h) < 0.2 * lineHeight) {
      const centerY = (g.minY + g.maxY) / 2;
      const char = centerY > lineBottom - 0.25 * lineHeight ? '.' : '*';
      symbols.push({ ...base, char, confidence: 1, candidates: [{ char, confidence: 1 }], pixels: null });
      continue;
    }

    const pixels = rasterizeStrokes28(g.strokes.map((i) => strokes[i]));
    const { probabilities } = model.network.forward(pixels);
    const bars = stackedBars(g, strokes);
    const flat = h < 0.3 * w && h < 0.45 * lineHeight;
    const allowed = bars === 2 ? EQUALS : bars === 3 ? TRIPLE_BAR : flat ? FLAT_SYMBOLS : undefined;
    const candidates = topCandidates(probabilities, model.classes, allowed);
    symbols.push({ ...base, char: candidates[0].char, confidence: candidates[0].confidence, candidates, pixels });
  }

  const { layout, expression, latex } = layoutSymbols(symbols);
  return { symbols, expression, latex, layout };
}

/** How a recognized symbol is shown to the user ('*' is the multiplication dot). */
export function displayChar(char: string): string {
  return char === '*' ? '·' : char;
}
