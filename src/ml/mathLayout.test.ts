import { describe, expect, it } from 'vitest';
import { layoutSymbols, RecognizedSymbol } from './mathLayout';

/** A symbol at x, y with width w and height h (the network is not involved: only the geometry). */
const sym = (id: number, char: string, x: number, y: number, w: number, h: number): RecognizedSymbol => ({
  id,
  char,
  confidence: 1,
  candidates: [{ char, confidence: 1 }],
  box: { x, y, width: w, height: h },
  pixels: null,
  strokeIndices: [id],
  role: 'base',
});

/** Digits and letters are 40 high and 24 wide on a line whose top is y = 100. */
const glyph = (id: number, char: string, x: number) => sym(id, char, x, 100, 24, 40);

describe('layoutSymbols', () => {
  it('reads symbols left to right', () => {
    const r = layoutSymbols([glyph(0, '2', 0), glyph(1, '4', 30), sym(2, '+', 60, 112, 24, 16), glyph(3, '1', 90)]);
    expect(r.expression).toBe('24+1');
  });

  it('finds exponents and indices', () => {
    const r = layoutSymbols([glyph(0, 'x', 0), sym(1, '2', 28, 86, 14, 22), sym(2, '-', 50, 118, 20, 3), glyph(3, 'y', 76), sym(4, 'n', 104, 128, 12, 18)]);
    expect(r.expression).toBe('x^(2)-y_(n)');
    expect(r.latex).toBe('x^{2} - y_{n}');
  });

  it('keeps long exponents together and returns to the line afterwards', () => {
    const r = layoutSymbols([glyph(0, '2', 0), sym(1, '1', 28, 86, 12, 22), sym(2, '0', 44, 86, 14, 22), glyph(3, '3', 66)]);
    expect(r.expression).toBe('2^(10)3');
  });

  it('does not take the minus or equals sign after x for an exponent or index', () => {
    const r = layoutSymbols([glyph(0, 'x', 0), sym(1, '-', 30, 118, 20, 3), glyph(2, '4', 56), sym(3, '=', 86, 112, 24, 14), glyph(4, '1', 116)]);
    expect(r.expression).toBe('x-4=1');
  });

  it('reads a fraction from its bar', () => {
    const r = layoutSymbols([sym(0, '1', 10, 60, 14, 30), sym(1, '-', 0, 100, 40, 3), sym(2, '2', 8, 112, 24, 30), sym(3, '+', 50, 92, 24, 20), sym(4, '3', 84, 80, 20, 30)]);
    expect(r.expression).toBe('\\frac(1,2)+3');
    expect(r.latex).toBe('\\frac{1}{2} + 3');
  });

  it('keeps a plain minus a minus', () => {
    expect(layoutSymbols([glyph(0, '5', 0), sym(1, '-', 30, 118, 22, 3), glyph(2, '3', 58)]).expression).toBe('5-3');
  });

  it('reads nested fractions', () => {
    const r = layoutSymbols([
      sym(0, 'a', 40, 20, 20, 30),
      sym(1, '-', 0, 60, 110, 3), // wide bar
      sym(2, 'b', 10, 74, 20, 30),
      sym(3, '-', 40, 112, 50, 3), // small bar under b ... as the denominator of the denominator
      sym(4, 'c', 55, 124, 20, 30),
    ]);
    expect(r.expression.startsWith('\\frac(a,')).toBe(true);
  });

  it('reads a square root with what is under its overline', () => {
    const r = layoutSymbols([sym(0, '√', 0, 90, 80, 56), glyph(1, '1', 36), glyph(2, '6', 54), sym(3, '=', 96, 112, 24, 14)]);
    expect(r.expression).toBe('\\sqrt(16)=');
    expect(r.latex).toBe('\\sqrt{1 6} =');
  });

  it('reads a sum with limits above and below', () => {
    const r = layoutSymbols([
      sym(0, 'Σ', 0, 80, 40, 70),
      sym(1, '4', 10, 40, 16, 30), // above
      sym(2, 'k', 0, 158, 14, 26),
      sym(3, '=', 16, 168, 14, 10),
      sym(4, '1', 34, 158, 10, 26),
      glyph(5, 'k', 56),
      sym(6, '2', 84, 88, 14, 22),
    ]);
    expect(r.expression).toBe('Σ_(k=1)^(4)k^(2)');
    expect(r.latex).toBe('\\sum_{k = 1}^{4} k^{2}');
  });

  it('reads an integral with limits at its right', () => {
    const r = layoutSymbols([
      sym(0, '∫', 0, 70, 22, 80),
      sym(1, '2', 28, 66, 12, 20),
      sym(2, '0', 20, 136, 12, 20),
      glyph(3, 'x', 70),
      glyph(4, 'd', 110),
      glyph(5, 'x', 140),
    ]);
    expect(r.expression).toBe('∫_(0)^(2)xdx');
  });

  it('tells the letter x from the multiplication sign by its neighbours', () => {
    expect(layoutSymbols([glyph(0, '2', 0), glyph(1, '×', 30), glyph(2, '+', 60)]).expression).toBe('2x+');
    expect(layoutSymbols([glyph(0, '3', 0), glyph(1, 'x', 30), glyph(2, '4', 60)]).expression).toBe('3×4');
  });

  it('recognizes log and ln when the l was read as a 1', () => {
    expect(layoutSymbols([glyph(0, '1', 0), glyph(1, 'o', 30), glyph(2, 'g', 60), glyph(3, 'x', 90)]).expression).toBe('logx');
    expect(layoutSymbols([glyph(0, '|', 0), glyph(1, 'n', 30), glyph(2, '(', 60)]).expression).toBe('ln(');
  });

  it('puts ≤ together from < and a bar under it', () => {
    const r = layoutSymbols([glyph(0, 'x', 0), sym(1, '<', 30, 100, 22, 28), sym(2, '-', 30, 132, 22, 3), glyph(3, '4', 60)]);
    expect(r.expression).toBe('x≤4');
  });

  it('tags the role of each symbol', () => {
    const symbols = [glyph(0, 'x', 0), sym(1, '2', 28, 86, 14, 22)];
    layoutSymbols(symbols);
    expect(symbols.map((s) => s.role)).toEqual(['base', 'sup']);
  });
});

describe('a page that defines and differentiates swish', () => {
  // swish(x) = x σ(β x)
  const definition = layoutSymbols([
    glyph(0, 's', 0), glyph(1, 'w', 26), glyph(2, 'i', 52), glyph(3, 's', 78), glyph(4, 'h', 104),
    sym(5, '(', 130, 96, 10, 48), glyph(6, 'x', 142), sym(7, ')', 168, 96, 10, 48),
    sym(8, '=', 190, 112, 24, 14),
    glyph(9, 'x', 224), glyph(10, 'σ', 252), sym(11, '(', 280, 96, 10, 48), glyph(12, 'β', 292), glyph(13, 'x', 320), sym(14, ')', 346, 96, 10, 48),
  ]);

  // ∂y over ∂x
  const partial = layoutSymbols([
    glyph(0, '∂', 10), glyph(1, 'y', 36),
    sym(2, '-', 0, 150, 80, 3),
    glyph(3, '∂', 10 + 0), glyph(4, 'x', 36),
  ].map((s, i) => (i >= 3 ? { ...s, box: { ...s.box, y: 165 } } : s)));

  it('reads σ( and the fraction of partial derivatives', () => {
    expect(definition.expression).toBe('swish(x)=xσ(βx)');
    expect(partial.expression).toBe('\\frac(∂y,∂x)');
  });
});
