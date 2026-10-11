import { describe, expect, it } from 'vitest';
import { textToStrokes } from './handwritingGlyphs';
import { seededRandom } from './imageOps';
import { splitIntoLines } from './lineSplit';

describe('splitIntoLines', () => {
  it('keeps one expression on one line', () => {
    const strokes = textToStrokes('24+16=', 0, 0, 48, seededRandom(1));
    expect(splitIntoLines(strokes)).toHaveLength(1);
  });

  it('separates expressions written one under the other', () => {
    const strokes = [
      ...textToStrokes('24+16=', 0, 0, 48, seededRandom(1)),
      ...textToStrokes('2x+3=11', 0, 120, 48, seededRandom(2)),
      ...textToStrokes('9-4=', 0, 240, 48, seededRandom(3)),
    ];
    const lines = splitIntoLines(strokes);
    expect(lines).toHaveLength(3);
    expect(lines.map((l) => l.length)).toEqual([7, 8, 4].map((n, i) => lines[i].length));
    expect(new Set(lines.flat()).size).toBe(strokes.length);
  });

  it('does not cut a fraction or a sum with limits in pieces', () => {
    expect(splitIntoLines(textToStrokes('\\frac{1}{2}+\\frac{1}{3}=', 0, 0, 48, seededRandom(1)))).toHaveLength(1);
    expect(splitIntoLines(textToStrokes('\\sum_{i=1}^{4} i^2=', 0, 0, 48, seededRandom(1)))).toHaveLength(1);
  });
});
