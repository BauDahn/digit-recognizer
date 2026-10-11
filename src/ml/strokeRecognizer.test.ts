import { readFileSync } from 'node:fs';
import path from 'node:path';
import { describe, expect, it } from 'vitest';
import { textToStrokes } from './handwritingGlyphs';
import { seededRandom } from './imageOps';
import { solve } from './mathSolver';
import { fromModelFile, ModelFile } from './modelFile';
import { groupStrokes, recognizeStrokes } from './strokeRecognizer';

const model = fromModelFile(JSON.parse(readFileSync(path.resolve(import.meta.dirname, '../../public/models/math.json'), 'utf8')) as ModelFile);
const read = (text: string, seed: number) => recognizeStrokes(textToStrokes(text, 0, 0, 48, seededRandom(seed)), model);

describe('groupStrokes', () => {
  it('groups the strokes of "+" and "=" into one symbol each', () => {
    const strokes = textToStrokes('1+2=', 0, 0, 50, seededRandom(1));
    expect(strokes).toHaveLength(6);
    expect(groupStrokes(strokes)).toHaveLength(4);
  });

  it('keeps a root and the number under its overline apart', () => {
    expect(groupStrokes(textToStrokes('\\sqrt{16}', 0, 0, 48, seededRandom(2)))).toHaveLength(3);
  });
});

describe('recognizeStrokes (pre-trained math network)', () => {
  it.each(['24+16=', '2x+3=11', 'x^2-4', '(3+5)/2=', '1234567890'])('reads "%s"', (text) => {
    for (const seed of [1, 2, 3]) {
      expect(read(text, seed).expression).toBe(text.replace('^2', '^(2)'));
    }
  });

  it('reads and solves', () => {
    expect(solve(read('2x+3=11', 4).expression).answer).toBe('x = 4');
    expect(solve(read('24+16=', 4).expression).answer).toBe('40');
  });

  it('reads fractions and roots', () => {
    for (const seed of [1, 2, 3]) {
      expect(solve(read('\\frac{1}{2}+\\frac{1}{3}=', seed).expression).answer).toBe('0.8333333333');
      expect(solve(read('\\sqrt{16}=', seed).expression).answer).toBe('4');
    }
  });

  it('reads Greek, calculus and sets symbols as single symbols', () => {
    for (const [text, char] of [['Σ', 'Σ'], ['∫', '∫'], ['∂', '∂'], ['∞', '∞']] as const) {
      expect(read(text, 1).symbols.map((s) => s.char)).toEqual([char]);
    }
  });

  it('reads a tall "=" (two bars with a wide gap) as "=", not as a digit', () => {
    const strokes = [
      ...textToStrokes('24+16', 0, 0, 70, seededRandom(1)),
      [{ x: 300, y: 20 }, { x: 330, y: 20 }],
      [{ x: 300, y: 45 }, { x: 330, y: 45 }],
    ];
    expect(recognizeStrokes(strokes, model).expression).toBe('24+16=');
  });

  it('detects small dots by size', () => {
    const strokes = [
      ...textToStrokes('3', 0, 0, 48, seededRandom(1)),
      [{ x: 40, y: 22 }, { x: 42, y: 23 }],
      ...textToStrokes('4', 55, 0, 48, seededRandom(1)),
    ];
    expect(recognizeStrokes(strokes, model).expression).toBe('3*4');
  });
});
