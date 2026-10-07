import { Box, rasterizeStrokes28, StrokePoint } from './imageOps';
import { LoadedModel } from './modelFile';

/**
 * Turns handwritten strokes from the math canvas into a line of text such as "2x^2+3=11":
 *
 * 1. Group strokes into symbols: strokes that overlap horizontally belong together
 *    ('=', '+', 'x' and many '4's are drawn with two strokes).
 * 2. Small marks are dots: '.' on the baseline, '·' (multiplication) in the middle.
 * 3. Every other symbol is rasterized to 28x28 like an MNIST image and classified by the network.
 * 4. Symbols that are raised and smaller than the previous one are exponents.
 */

export interface RecognizedSymbol {
  char: string;
  confidence: number;
  candidates: Array<{ char: string; confidence: number }>;
  box: Box;
  pixels: Float32Array | null; // what the network saw (null for geometric dots)
  isSuperscript: boolean;
  strokeIndices: number[];
}

export interface StrokeRecognitionResult {
  symbols: RecognizedSymbol[];
  expression: string;
}

interface Group {
  strokes: number[];
  minX: number;
  maxX: number;
  minY: number;
  maxY: number;
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

export function groupStrokes(strokes: StrokePoint[][]): Group[] {
  const order = strokes
    .map((points, i) => ({ i, ...strokeBox(points) }))
    .filter((s) => Number.isFinite(s.minX))
    .sort((a, b) => a.minX - b.minX);

  const groups: Group[] = [];
  for (const s of order) {
    const sw = Math.max(1, s.maxX - s.minX);
    const target = groups.find((g) => {
      const overlap = Math.min(g.maxX, s.maxX) - Math.max(g.minX, s.minX);
      const gw = Math.max(1, g.maxX - g.minX);
      if (overlap > 0.45 * Math.min(sw, gw)) return true;
      // A thin stroke (the vertical bar of '+' or '4') has almost no width: merge it
      // when its center falls inside the wider group.
      const [narrow, wide] = sw < gw ? [s, g] : [g, s];
      const center = (narrow.minX + narrow.maxX) / 2;
      return center > wide.minX && center < wide.maxX;
    });
    if (target) {
      target.strokes.push(s.i);
      target.minX = Math.min(target.minX, s.minX);
      target.maxX = Math.max(target.maxX, s.maxX);
      target.minY = Math.min(target.minY, s.minY);
      target.maxY = Math.max(target.maxY, s.maxY);
    } else {
      groups.push({ strokes: [s.i], minX: s.minX, maxX: s.maxX, minY: s.minY, maxY: s.maxY });
    }
  }
  return groups.sort((a, b) => a.minX - b.minX);
}

function median(values: number[]): number {
  if (values.length === 0) return 0;
  const sorted = [...values].sort((a, b) => a - b);
  return sorted[Math.floor(sorted.length / 2)];
}

function topCandidates(probabilities: Float32Array, classes: string[], allowed?: Set<string>, n = 3) {
  return classes
    .map((char, i) => ({ char, confidence: probabilities[i] }))
    .filter((c) => !allowed || allowed.has(c.char))
    .sort((a, b) => b.confidence - a.confidence)
    .slice(0, n);
}

const FLAT_SYMBOLS = new Set(['-', '=']);
const EQUALS = new Set(['=']);

/** Two flat bars, one above the other without touching: '=' however wide the gap between them. */
function isTwoBars(g: Group, strokes: StrokePoint[][]): boolean {
  if (g.strokes.length !== 2) return false;
  const [a, b] = g.strokes.map((i) => strokeBox(strokes[i])).sort((p, q) => p.minY - q.minY);
  const flat = (s: typeof a) => s.maxY - s.minY < 0.35 * Math.max(1, s.maxX - s.minX);
  return flat(a) && flat(b) && a.maxY < b.minY;
}
const CAN_HAVE_EXPONENT = /[0-9x)]/;

export function recognizeStrokes(strokes: StrokePoint[][], model: LoadedModel): StrokeRecognitionResult {
  const groups = groupStrokes(strokes);
  if (groups.length === 0) return { symbols: [], expression: '' };

  // Typical symbol height, ignoring flat symbols and dots
  const heights = groups.map((g) => g.maxY - g.minY);
  const lineHeight = Math.max(1, median(heights.filter((h) => h > 0.4 * Math.max(...heights))));
  const lineTop = median(groups.map((g) => g.minY));
  const lineBottom = lineTop + lineHeight;

  const symbols: RecognizedSymbol[] = [];
  for (const g of groups) {
    const w = g.maxX - g.minX;
    const h = g.maxY - g.minY;
    const box = { x: g.minX, y: g.minY, width: w, height: h };
    const base = { box, strokeIndices: g.strokes, isSuperscript: false };

    if (Math.max(w, h) < 0.2 * lineHeight) {
      const centerY = (g.minY + g.maxY) / 2;
      const char = centerY > lineBottom - 0.25 * lineHeight ? '.' : '*';
      symbols.push({ ...base, char, confidence: 1, candidates: [{ char, confidence: 1 }], pixels: null });
      continue;
    }

    const pixels = rasterizeStrokes28(g.strokes.map((i) => strokes[i]));
    const { probabilities } = model.network.forward(pixels);
    // A very flat shape can only be '-' or '=': the 28x28 image loses the size information.
    const isFlat = h < 0.3 * w && h < 0.45 * lineHeight;
    const allowed = isTwoBars(g, strokes) ? EQUALS : isFlat ? FLAT_SYMBOLS : undefined;
    const candidates = topCandidates(probabilities, model.classes, allowed);
    symbols.push({ ...base, char: candidates[0].char, confidence: candidates[0].confidence, candidates, pixels });
  }

  // Exponents: smaller symbols whose bottom is above the middle of the previous symbol
  for (let i = 1; i < symbols.length; i++) {
    const prev = symbols[i - 1];
    const cur = symbols[i];
    if (cur.pixels === null || FLAT_SYMBOLS.has(cur.char) || !/[0-9x(]/.test(cur.char)) continue;
    // Compare against the base symbol, also for the 2nd, 3rd... symbol of an exponent
    let baseIndex = i - 1;
    while (baseIndex > 0 && symbols[baseIndex].isSuperscript) baseIndex--;
    const anchor = symbols[baseIndex];
    const raised = cur.box.y + cur.box.height < anchor.box.y + anchor.box.height * 0.6;
    const smaller = cur.box.height < anchor.box.height * 0.9 || cur.box.height < lineHeight * 0.7;
    if (prev.isSuperscript && raised) cur.isSuperscript = true;
    else if (!prev.isSuperscript && raised && smaller && CAN_HAVE_EXPONENT.test(prev.char)) cur.isSuperscript = true;
  }

  return { symbols, expression: buildExpression(symbols) };
}

/** Join the symbols into text the math parser understands, wrapping exponents in ^( ). */
export function buildExpression(symbols: Pick<RecognizedSymbol, 'char' | 'isSuperscript'>[]): string {
  let out = '';
  for (let i = 0; i < symbols.length; i++) {
    const s = symbols[i];
    const prevSup = i > 0 && symbols[i - 1].isSuperscript;
    if (s.isSuperscript && !prevSup) out += '^(';
    if (!s.isSuperscript && prevSup) out += ')';
    out += s.char;
  }
  if (symbols.length > 0 && symbols[symbols.length - 1].isSuperscript) out += ')';
  return out;
}

/** How a recognized symbol is shown to the user ('*' is the multiplication dot). */
export function displayChar(char: string): string {
  return char === '*' ? '·' : char;
}
