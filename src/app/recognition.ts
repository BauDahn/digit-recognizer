import { splitIntoLines } from '../ml/lineSplit';
import { layoutSymbols, RecognizedSymbol } from '../ml/mathLayout';
import { LoadedModel } from '../ml/modelFile';
import { collectDefinitions, solve, SolveResult } from '../ml/mathSolver';
import { recognizeStrokes } from '../ml/strokeRecognizer';
import { Stroke } from '../types';

/** One line of handwritten math: what was read, how it was laid out and what it solves to. */
export interface LineResult {
  /** Identifies the line across re-solves (the answer strokes written next to it carry this key) */
  key: string;
  strokeIds: string[];
  symbols: RecognizedSymbol[];
  /** Text for the parser (what the symbols mean) */
  expression: string;
  /** LaTeX of what was read, even when it cannot be computed */
  readLatex: string;
  solved: SolveResult;
  bounds: { x: number; y: number; width: number; height: number };
}

function boundsOf(symbols: RecognizedSymbol[]) {
  if (symbols.length === 0) return { x: 0, y: 0, width: 0, height: 0 };
  const x = Math.min(...symbols.map((s) => s.box.x));
  const y = Math.min(...symbols.map((s) => s.box.y));
  return {
    x,
    y,
    width: Math.max(...symbols.map((s) => s.box.x + s.box.width)) - x,
    height: Math.max(...symbols.map((s) => s.box.y + s.box.height)) - y,
  };
}

/**
 * Solve every line again, now that all the lines are known: "y = swish(x)" on one line is what
 * "∂y/∂x" means on another. `context` are lines read earlier that are not being read now.
 */
export function resolveLines(lines: LineResult[], context: string[] = []): LineResult[] {
  const defs = collectDefinitions([...context, ...lines.map((l) => l.expression)]);
  return lines.map((line) => ({ ...line, solved: solve(line.expression, defs) }));
}

/** Recognize the pen strokes of a page, one result per line of math. */
export function recognizePage(strokes: Stroke[], model: LoadedModel, splitLines = true, context: string[] = []): LineResult[] {
  const pen = strokes.filter((s) => s.tool === 'pen' && s.points.length > 0 && !s.answerFor);
  if (pen.length === 0) return [];
  const groups = splitLines ? splitIntoLines(pen.map((s) => s.points)) : [pen.map((_, i) => i)];

  const lines = groups
    .map((indices) => {
      const lineStrokes = indices.map((i) => pen[i]);
      const recognition = recognizeStrokes(
        lineStrokes.map((s) => s.points),
        model
      );
      // strokeIndices refer to positions in lineStrokes: keep them valid for the caller
      const result: LineResult = {
        key: `line-${lineStrokes[0].id}-${lineStrokes.length}`,
        strokeIds: lineStrokes.map((s) => s.id),
        symbols: recognition.symbols,
        expression: recognition.expression,
        readLatex: recognition.latex,
        solved: solve(recognition.expression),
        bounds: boundsOf(recognition.symbols),
      };
      return result;
    })
    .filter((line) => line.symbols.length > 0)
    .sort((a, b) => a.bounds.y - b.bounds.y);
  return resolveLines(lines, context);
}

/** After the user changed a symbol: lay the line out again and solve it again. */
export function relayoutLine(line: LineResult): LineResult {
  const { expression, latex } = layoutSymbols(line.symbols);
  return { ...line, expression, readLatex: latex, solved: solve(expression), bounds: boundsOf(line.symbols) };
}
