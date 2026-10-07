import { readFileSync } from 'node:fs';
import path from 'node:path';
import { describe, expect, it } from 'vitest';
import { textToStrokes } from './handwritingGlyphs';
import { seededRandom } from './imageOps';
import { solve } from './mathParser';
import { fromModelFile, ModelFile } from './modelFile';
import { buildExpression, groupStrokes, recognizeStrokes } from './strokeRecognizer';

const model = fromModelFile(
  JSON.parse(readFileSync(path.resolve(import.meta.dirname, '../../public/models/symbols.json'), 'utf8')) as ModelFile
);

describe('groupStrokes', () => {
  it('groups the two strokes of "+" and "=" into one symbol each', () => {
    const strokes = textToStrokes('1+2=', 0, 0, 50, seededRandom(1));
    expect(strokes).toHaveLength(6);
    expect(groupStrokes(strokes)).toHaveLength(4);
  });
});

describe('buildExpression', () => {
  it('wraps raised symbols in ^( )', () => {
    const sym = (char: string, isSuperscript = false) => ({ char, isSuperscript });
    expect(buildExpression([sym('x'), sym('1', true), sym('0', true), sym('+'), sym('1')])).toBe('x^(10)+1');
  });
});

describe('recognizeStrokes (pre-trained symbols network)', () => {
  it.each(['24+16=', '2x+3=11', 'x^2-4', '(3+5)/2=', '1234567890'])('reads "%s"', (text) => {
    for (const seed of [1, 2, 3]) {
      const { expression } = recognizeStrokes(textToStrokes(text, 0, 0, 48, seededRandom(seed)), model);
      expect(expression).toBe(text.replace('^2', '^(2)'));
    }
  });

  it('solves what it reads', () => {
    const { expression } = recognizeStrokes(textToStrokes('2x+3=11', 0, 0, 48, seededRandom(4)), model);
    expect(solve(expression).answer).toBe('x = 4');
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
