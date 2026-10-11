/**
 * Page layout of handwritten math. Given the recognized symbols and where they sit on the page,
 * work out what is a fraction, a square root, a sum with limits, an exponent or an index,
 * and write the result as LaTeX (to show it) and as a line of text (to compute it).
 *
 * The geometry rules work from the bounding boxes only, so they also apply after the user
 * corrects a symbol: the layout is simply recomputed.
 */
import { Box } from './imageOps';
import { symbolLatex } from './symbols';

export type SymbolRole = 'base' | 'sup' | 'sub' | 'numerator' | 'denominator' | 'radicand' | 'root-index' | 'upper-limit' | 'lower-limit';

export interface RecognizedSymbol {
  id: number;
  char: string;
  confidence: number;
  candidates: Array<{ char: string; confidence: number }>;
  box: Box;
  /** What the network saw (null for dots, which are told apart by size alone). */
  pixels: Float32Array | null;
  strokeIndices: number[];
  /** Where the symbol sits in the formula, filled in by layoutSymbols. */
  role: SymbolRole;
  /** The user fixed this symbol by hand. */
  corrected?: boolean;
}

export type LayoutNode =
  | { type: 'sym'; char: string; id: number }
  | { type: 'row'; items: LayoutNode[] }
  | { type: 'frac'; num: LayoutNode; den: LayoutNode }
  | { type: 'sqrt'; body: LayoutNode; index?: LayoutNode }
  | { type: 'script'; base: LayoutNode; sup?: LayoutNode; sub?: LayoutNode }
  | { type: 'bigop'; op: LayoutNode; lower?: LayoutNode; upper?: LayoutNode };

interface Item {
  box: Box;
  node: LayoutNode;
  /** Set for single symbols */
  char?: string;
  symbolId?: number;
}

const cx = (b: Box) => b.x + b.width / 2;
const cy = (b: Box) => b.y + b.height / 2;
const right = (b: Box) => b.x + b.width;
const bottom = (b: Box) => b.y + b.height;

function unionBox(boxes: Box[]): Box {
  const x = Math.min(...boxes.map((b) => b.x));
  const y = Math.min(...boxes.map((b) => b.y));
  return { x, y, width: Math.max(...boxes.map(right)) - x, height: Math.max(...boxes.map(bottom)) - y };
}

function median(values: number[]): number {
  if (values.length === 0) return 0;
  const sorted = [...values].sort((a, b) => a - b);
  return sorted[Math.floor(sorted.length / 2)];
}

const BIG_OPERATORS = new Set(['Σ', '∑', '∏', '∫', '∮']);
/** Symbols that never carry an exponent or an index of their own. */
const NO_SCRIPTS = new Set([
  '+', '-', '×', '÷', '/', '±', '∘', '*', '=', '≠', '≈', '≡', '∼', '≅', '<', '>', '≤', '≥', '∝', '(', '[', '{', '⟨', '⌊', '⌈',
  '→', '←', '⇒', '⇐', '⇔', '↔', '↦', '.', ',', '∂', '∇', '∀', '∃', '∧', '∨', '¬', '∈', '∉', '⊂', '⊆', '⊃', '⊇', '∪', '∩', '∴', '∵',
]);

// ------------------------------------------------------------------ the layout

export interface LayoutResult {
  layout: LayoutNode;
  /** Text the math parser understands, e.g. "Σ_(i=1)^(5)i^(2)" */
  expression: string;
  latex: string;
}

export function layoutSymbols(symbols: RecognizedSymbol[]): LayoutResult {
  for (const s of symbols) s.role = 'base';
  const byId = new Map(symbols.map((s) => [s.id, s]));
  const items: Item[] = symbols.map((s) => ({
    box: s.box,
    node: { type: 'sym', char: s.char, id: s.id },
    char: s.char,
    symbolId: s.id,
  }));
  const lineHeight = estimateLineHeight(items);
  disambiguateTimes(symbols, lineHeight);
  fixFunctionNames(symbols);
  items.forEach((item, i) => {
    item.char = symbols[i].char;
    item.node = { type: 'sym', char: symbols[i].char, id: symbols[i].id };
  });
  const ctx: Ctx = { lineHeight, setRole: (id, role) => { const s = byId.get(id); if (s) s.role = role; } };
  const layout = layoutRow(items, ctx);
  return { layout, expression: toLinear(layout), latex: toLatexLayout(layout) };
}

/**
 * Height of a normal symbol: the 80th percentile of the heights of ordinary symbols (digits and
 * letters). Dots, bars, brackets, roots and big operators are not ordinary: they are taller or
 * shorter on purpose.
 */
function estimateLineHeight(items: Item[]): number {
  const unusual = (i: Item) => isTiny(i) || i.box.height <= 0.3 * i.box.width || (i.char !== undefined && (BIG_OPERATORS.has(i.char) || '√()[]{}⟨⟩⌊⌋⌈⌉|‖∫'.includes(i.char)));
  const ordinary = items.filter((i) => !unusual(i)).map((i) => i.box.height).sort((a, b) => a - b);
  if (ordinary.length > 0) return Math.max(1, ordinary[Math.floor(0.8 * (ordinary.length - 1) + 0.5)]);
  return Math.max(1, median(items.map((i) => i.box.height)));
}

/**
 * The network cannot tell the letter x from the multiplication sign ×: they are the same shape.
 * It is a multiplication when it stands between two numbers (or brackets) on the same line,
 * and the variable x otherwise ("2x + 3", "x²").
 */
function disambiguateTimes(symbols: RecognizedSymbol[], lineHeight: number) {
  const sorted = [...symbols].sort((a, b) => cx(a.box) - cx(b.box));
  const isOperandEnd = (c: string) => /[0-9]/.test(c) || c === ')' || c === ']';
  const isOperandStart = (c: string) => /[0-9]/.test(c) || c === '(' || c === '[';
  sorted.forEach((sym, i) => {
    if (sym.corrected || (sym.char !== 'x' && sym.char !== '×')) return;
    const prev = sorted[i - 1];
    const next = sorted[i + 1];
    const onLine = (other: RecognizedSymbol) => Math.abs(cy(other.box) - cy(sym.box)) < 0.45 * lineHeight;
    const between = !!prev && !!next && isOperandEnd(prev.char) && isOperandStart(next.char) && onLine(prev) && onLine(next);
    sym.char = between ? '×' : 'x';
  });
}

/**
 * A lone vertical bar is a 1, an l or a | to the network. Before "og", "n" or "im" it is the l of
 * log, ln or lim.
 */
function fixFunctionNames(symbols: RecognizedSymbol[]) {
  const sorted = [...symbols].sort((a, b) => cx(a.box) - cx(b.box));
  const bar = (s?: RecognizedSymbol) => !!s && !s.corrected && /^[1|lI↑ℓ]$/.test(s.char);
  const is = (s: RecognizedSymbol | undefined, c: string) => !!s && s.char === c;
  sorted.forEach((s, i) => {
    if (!bar(s)) return;
    const [a, b] = [sorted[i + 1], sorted[i + 2]];
    if ((is(a, 'o') && is(b, 'g')) || (is(a, 'i') && is(b, 'm')) || (is(a, 'n') && !/[0-9]/.test(b?.char ?? ''))) s.char = 'l';
  });
}

interface Ctx {
  lineHeight: number;
  setRole: (id: number, role: SymbolRole) => void;
}

function isTiny(item: Item): boolean {
  return item.char === '.' || item.char === '*';
}

/** Mark every symbol inside `node` with a role (the outermost role wins, set first). */
function assignRole(node: LayoutNode, role: SymbolRole, ctx: Ctx) {
  switch (node.type) {
    case 'sym':
      ctx.setRole(node.id, role);
      break;
    case 'row':
      node.items.forEach((n) => assignRole(n, role, ctx));
      break;
    default:
      break; // nested structure keeps the roles it got when it was built
  }
}

function layoutRow(itemsIn: Item[], ctx: Ctx): LayoutNode {
  let items = [...itemsIn];
  if (items.length === 0) return { type: 'row', items: [] };

  items = extractFractions(items, ctx);
  items = extractRadicals(items, ctx);
  items = extractBigOperators(items, ctx);
  items = mergeChevronBars(items);

  items.sort((a, b) => cx(a.box) - cx(b.box));
  return foldScripts(items, ctx);
}

function isBarCandidate(item: Item): boolean {
  return item.char === '-' && item.box.width > item.box.height * 1.5;
}

/** Fraction bars: a flat '-' with something above it and something below it. */
function extractFractions(items: Item[], ctx: Ctx): Item[] {
  const bars = items.filter(isBarCandidate).sort((a, b) => b.box.width - a.box.width);
  const remaining = new Set(items);
  const created: Item[] = [];
  const reach = 2.4 * ctx.lineHeight;

  for (const bar of bars) {
    if (!remaining.has(bar)) continue;
    const from = bar.box.x - 0.15 * bar.box.width;
    const to = right(bar.box) + 0.15 * bar.box.width;
    const candidates = [...remaining].filter((it) => {
      if (it === bar || isBarCandidate(it) && it.box.width >= bar.box.width) return false;
      if (it.box.x < bar.box.x - 0.2 * bar.box.width && right(it.box) > right(bar.box) + 0.2 * bar.box.width) return false; // encloses the bar (a root)
      return cx(it.box) >= from && cx(it.box) <= to;
    });
    const above = candidates.filter((it) => cy(it.box) < cy(bar.box) && bar.box.y - bottom(it.box) < reach);
    const below = candidates.filter((it) => cy(it.box) > cy(bar.box) && it.box.y - bottom(bar.box) < reach);
    if (above.length === 0 || below.length === 0) continue;

    const num = layoutRow(above, ctx);
    const den = layoutRow(below, ctx);
    assignRole(num, 'numerator', ctx);
    assignRole(den, 'denominator', ctx);
    ctx.setRole(bar.symbolId!, 'base');
    [bar, ...above, ...below].forEach((it) => remaining.delete(it));
    created.push({ box: unionBox([bar.box, ...above.map((i) => i.box), ...below.map((i) => i.box)]), node: { type: 'frac', num, den } });
  }
  return [...remaining, ...created];
}

/** √ with whatever sits under its overline (and a small index to its upper left for cube roots). */
function extractRadicals(items: Item[], ctx: Ctx): Item[] {
  const roots = items.filter((it) => it.char === '√').sort((a, b) => b.box.width - a.box.width);
  const remaining = new Set(items);
  const created: Item[] = [];

  for (const root of roots) {
    const { x, y, width: w, height: h } = root.box;
    const others = [...remaining].filter((it) => it !== root);
    const inside = (it: Item) => cx(it.box) > x + 0.2 * w && cx(it.box) < x + w * 1.04 && cy(it.box) > y - 0.1 * h && cy(it.box) < y + h * 1.05;
    let radicand = others.filter(inside);

    // A √ drawn without its overline: take what follows, written at the same height
    if (radicand.length === 0 && w < 1.3 * h) {
      const following = others.filter((it) => cx(it.box) > x + w * 0.6 && cy(it.box) > y - 0.1 * h && cy(it.box) < y + h * 1.05).sort((a, b) => a.box.x - b.box.x);
      let edge = right(root.box);
      radicand = [];
      for (const it of following) {
        if (it.box.x - edge > 0.9 * ctx.lineHeight) break;
        radicand.push(it);
        edge = Math.max(edge, right(it.box));
      }
    }
    if (radicand.length === 0) continue;

    const indexItems = others.filter((it) => !radicand.includes(it) && cx(it.box) < x + 0.3 * w && cx(it.box) > x - 0.8 * h && cy(it.box) < y + 0.5 * h && it.box.height < 0.6 * h);
    const body = layoutRow(radicand, ctx);
    assignRole(body, 'radicand', ctx);
    let index: LayoutNode | undefined;
    if (indexItems.length > 0) {
      index = layoutRow(indexItems, ctx);
      assignRole(index, 'root-index', ctx);
    }
    [root, ...radicand, ...indexItems].forEach((it) => remaining.delete(it));
    created.push({
      box: unionBox([root.box, ...radicand.map((i) => i.box), ...indexItems.map((i) => i.box)]),
      node: { type: 'sqrt', body, index },
    });
  }
  return [...remaining, ...created];
}

/** Σ, ∏ and ∫: what is written above and below them are the limits. */
function extractBigOperators(items: Item[], ctx: Ctx): Item[] {
  const ops = items.filter((it) => it.char !== undefined && BIG_OPERATORS.has(it.char)).sort((a, b) => a.box.x - b.box.x);
  const remaining = new Set(items);
  const created: Item[] = [];

  for (const op of ops) {
    const { x, y, width: w, height: h } = op.box;
    const isIntegral = op.char === '∫' || op.char === '∮';
    const from = x - 0.3 * w;
    const to = right(op.box) + (isIntegral ? 1.2 * w : 0.3 * w);
    const near = [...remaining].filter((it) => it !== op && cx(it.box) >= from && cx(it.box) <= to && it.box.height < 0.85 * h);
    const upper = near.filter((it) => cy(it.box) < y + 0.25 * h);
    const lower = near.filter((it) => cy(it.box) > y + 0.75 * h);
    if (upper.length === 0 && lower.length === 0) continue;

    const upperNode = upper.length ? layoutRow(upper, ctx) : undefined;
    const lowerNode = lower.length ? layoutRow(lower, ctx) : undefined;
    if (upperNode) assignRole(upperNode, 'upper-limit', ctx);
    if (lowerNode) assignRole(lowerNode, 'lower-limit', ctx);
    [op, ...upper, ...lower].forEach((it) => remaining.delete(it));
    created.push({ box: op.box, node: { type: 'bigop', op: op.node, upper: upperNode, lower: lowerNode } });
  }
  return [...remaining, ...created];
}

/** '<' or '>' with a bar right under it is '≤' or '≥' (and nothing is written below, or it would be a fraction). */
function mergeChevronBars(items: Item[]): Item[] {
  const result = [...items];
  for (const chev of items) {
    if (chev.char !== '<' && chev.char !== '>') continue;
    const bar = result.find(
      (it) =>
        isBarCandidate(it) &&
        it.box.y > cy(chev.box) &&
        it.box.y - bottom(chev.box) < 0.5 * chev.box.height &&
        Math.min(right(it.box), right(chev.box)) - Math.max(it.box.x, chev.box.x) > 0.6 * Math.min(it.box.width, chev.box.width)
    );
    if (!bar) continue;
    const char = chev.char === '<' ? '≤' : '≥';
    const merged: Item = { box: unionBox([chev.box, bar.box]), node: { type: 'sym', char, id: chev.symbolId! }, char, symbolId: chev.symbolId };
    result.splice(result.indexOf(chev), 1, merged);
    result.splice(result.indexOf(bar), 1);
  }
  return result;
}

type Relation = 'sup' | 'sub' | null;

/**
 * Is `it` an exponent or an index of `base`? An exponent floats high: its center is more than 0.62
 * line heights above the bottom of the base and in the base's top part. An index sits low and is
 * clearly smaller than a normal symbol. Everything else is on the line. Tall composite bases
 * (fractions, roots) only take scripts above their top or below their bottom edge.
 */
function scriptRelation(base: Item, it: Item, lineHeight: number): Relation {
  if (isTiny(it)) return null;
  const bh = Math.max(base.box.height, 1);
  const t = (cy(it.box) - base.box.y) / bh;
  if (base.node.type !== 'sym') {
    if (cy(it.box) < base.box.y && it.box.height <= 0.85 * lineHeight) return 'sup';
    if (cy(it.box) > bottom(base.box) && it.box.height <= 0.6 * lineHeight) return 'sub';
    return null;
  }
  const rise = (bottom(base.box) - cy(it.box)) / lineHeight;
  if (rise > 0.62 && t < 0.3 && it.box.height <= Math.max(0.85 * lineHeight, 0.9 * bh)) return 'sup';
  if (rise < 0.2 && t > 0.7 && it.box.height <= Math.max(0.6 * lineHeight, 0.7 * bh)) return 'sub';
  return null;
}

function foldScripts(sorted: Item[], ctx: Ctx): LayoutNode {
  const out: LayoutNode[] = [];
  let base: Item | null = null;
  let sup: Item[] = [];
  let sub: Item[] = [];
  let edge = 0; // right edge of the last item attached to the current base

  const flush = () => {
    if (!base) return;
    if (sup.length === 0 && sub.length === 0) {
      out.push(base.node);
    } else {
      const node: LayoutNode = { type: 'script', base: base.node };
      if (sup.length) {
        node.sup = layoutRow(sup, ctx);
        assignRole(node.sup, 'sup', ctx);
      }
      if (sub.length) {
        node.sub = layoutRow(sub, ctx);
        assignRole(node.sub, 'sub', ctx);
      }
      out.push(node);
    }
    base = null;
    sup = [];
    sub = [];
  };

  for (const it of sorted) {
    if (base) {
      const canCarry = base.char === undefined || !NO_SCRIPTS.has(base.char);
      const rel = canCarry && it.box.x - edge < 1.1 * ctx.lineHeight ? scriptRelation(base, it, ctx.lineHeight) : null;
      if (rel) {
        (rel === 'sup' ? sup : sub).push(it);
        edge = Math.max(edge, right(it.box));
        continue;
      }
    }
    flush();
    base = it;
    edge = right(it.box);
  }
  flush();
  return out.length === 1 ? out[0] : { type: 'row', items: out };
}

// ------------------------------------------------------------------ output

/** One line of text for the math parser: x_(1)^(2), \frac(a,b), \sqrt(a), Σ_(i=1)^(n). */
export function toLinear(node: LayoutNode): string {
  const group = (n: LayoutNode | undefined) => (n ? `(${toLinear(n)})` : '');
  switch (node.type) {
    case 'sym':
      return node.char;
    case 'row':
      return node.items.map(toLinear).join('');
    case 'frac':
      return `\\frac(${toLinear(node.num)},${toLinear(node.den)})`;
    case 'sqrt':
      return node.index ? `\\root(${toLinear(node.index)},${toLinear(node.body)})` : `\\sqrt(${toLinear(node.body)})`;
    case 'script':
      return `${toLinear(node.base)}${node.sub ? `_${group(node.sub)}` : ''}${node.sup ? `^${group(node.sup)}` : ''}`;
    case 'bigop':
      return `${toLinear(node.op)}${node.lower ? `_${group(node.lower)}` : ''}${node.upper ? `^${group(node.upper)}` : ''}`;
  }
}

const KNOWN_WORDS = ['arcsin', 'arccos', 'arctan', 'sin', 'cos', 'tan', 'cot', 'sec', 'csc', 'log', 'ln', 'exp', 'lim'];

export function toLatexLayout(node: LayoutNode): string {
  switch (node.type) {
    case 'sym':
      return symbolLatex(node.char);
    case 'row': {
      const parts: string[] = [];
      for (let i = 0; i < node.items.length; i++) {
        // Letters that spell sin, cos, ln ... are a function name, not a product of variables
        const word = KNOWN_WORDS.find((w) => w.split('').every((ch, k) => {
          const n = node.items[i + k];
          return n && n.type === 'sym' && n.char === ch;
        }));
        if (word) {
          parts.push(`\\${word}\\,`);
          i += word.length - 1;
        } else {
          parts.push(toLatexLayout(node.items[i]));
        }
      }
      return parts.join(' ');
    }
    case 'frac':
      return `\\frac{${toLatexLayout(node.num)}}{${toLatexLayout(node.den)}}`;
    case 'sqrt':
      return node.index ? `\\sqrt[${toLatexLayout(node.index)}]{${toLatexLayout(node.body)}}` : `\\sqrt{${toLatexLayout(node.body)}}`;
    case 'script': {
      const base = toLatexLayout(node.base);
      const braced = /^\\?[A-Za-z0-9]+$/.test(base) || node.base.type === 'sym' ? base : `{${base}}`;
      return `${braced}${node.sub ? `_{${toLatexLayout(node.sub)}}` : ''}${node.sup ? `^{${toLatexLayout(node.sup)}}` : ''}`;
    }
    case 'bigop':
      return `${toLatexLayout(node.op)}${node.lower ? `_{${toLatexLayout(node.lower)}}` : ''}${node.upper ? `^{${toLatexLayout(node.upper)}}` : ''}`;
  }
}
