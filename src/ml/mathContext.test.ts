import { describe, expect, it } from 'vitest';
import { evaluate } from './mathAst';
import { differentiate, integrate } from './mathCalculus';
import { collectDefinitions } from './mathContext';
import { parseExpression } from './mathParser';
import { solve } from './mathSolver';

const page = (lines: string[]) => {
  const defs = collectDefinitions(lines);
  return lines.map((line) => solve(line, defs));
};

/** Central difference, to check a symbolic derivative against the numbers. */
const numericDerivative = (f: (x: number) => number, x: number, h = 1e-5) => (f(x + h) - f(x - h)) / (2 * h);

describe('definitions across lines', () => {
  const lines = ['swish(x)=xσ(βx)', 'y=swish(x)', '\\frac(∂y,∂x)='];

  it('differentiates y = swish(x) = xσ(βx) with respect to x', () => {
    const results = page(lines);
    const last = results[2];
    expect(last.kind).toBe('evaluation');
    expect(last.error).toBeUndefined();
    // the answer is a valid expression: σ(βx) + βxσ(βx)(1 − σ(βx))
    const derivative = parseExpression(last.answer!);
    const swish = (x: number, beta: number) => x / (1 + Math.exp(-beta * x));
    for (const [x, beta] of [[0.7, 1], [-1.2, 2.5], [2, 0.5]]) {
      expect(evaluate(derivative, { x, β: beta })).toBeCloseTo(numericDerivative((t) => swish(t, beta), x), 5);
    }
    expect(last.steps.join('\n')).toContain('Product rule');
    expect(last.steps.join('\n')).toContain('σ′(u) = σ(u)(1 − σ(u))');
  });

  it('does not care about the order of the lines', () => {
    const results = page([lines[2], lines[1], lines[0]]);
    expect(results[0].kind).toBe('evaluation');
  });

  it('shows the definitions as typeset text', () => {
    const results = page(lines);
    expect(results[0].latex).toContain('\\operatorname{swish}');
    expect(results[0].latex).toContain('\\sigma');
    expect(results[0].kind).toBe('symbolic');
  });

  it('says what is missing when y is not defined', () => {
    const result = solve('\\frac(∂y,∂x)=');
    expect(result.kind).toBe('symbolic');
    expect(result.steps[0]).toContain('not defined');
  });

  it('uses user functions in other expressions', () => {
    const results = page(['f(x)=x^2+1', 'f(3)=', '\\frac(d,dx)f(x)=']);
    expect(results[1].answer).toBe('10');
    expect(results[2].answer).toBe('2 · x');
  });

  it('plots a definition with a single unknown', () => {
    const results = page(['y=x^2-4']);
    expect(results[0].kind).toBe('function');
  });

  it('does not turn x = 5 into a definition', () => {
    const results = page(['x=5', '2x+3=13']);
    expect(results[1].answer).toBe('x = 5');
  });
});

describe('σ and friends', () => {
  it('reads σ( as the sigmoid', () => {
    expect(solve('σ(0)=').answer).toBe('0.5');
    expect(solve('swish(0)=').answer).toBe('0');
  });
  it('differentiates the activation functions', () => {
    const f = (name: string) => parseExpression(`${name}(x)`);
    for (const name of ['σ', 'swish', 'tanh', 'softplus', 'sinh', 'cosh']) {
      const expr = f(name);
      const d = differentiate(expr, 'x');
      for (const x of [-1.3, 0.4, 2]) {
        expect(evaluate(d, { x })).toBeCloseTo(numericDerivative((t) => evaluate(expr, { x: t }), x), 5);
      }
    }
  });
  it('differentiates partially, mixed and nested', () => {
    expect(solve('\\frac(∂,∂x)(x^2y+y^3)').answer).toBe('2 · x · y');
    expect(solve('\\frac(∂,∂y)(x^2y+y^3)').answer).toBe('x² + 3 · y²');
    expect(solve('\\frac(∂^2 f,∂x ∂y)').kind).toBe('symbolic');
    const mixed = page(['f=x^2y^3', '\\frac(∂^2 f,∂x∂y)=']);
    expect(mixed[1].answer).toBe('6 · x · y²');
  });
});

describe('integrals', () => {
  const check = (src: string, v = 'x') => {
    const f = parseExpression(src);
    const F = integrate(f, v);
    expect(F, src).not.toBeNull();
    for (const x of [0.4, 0.9, 1.6]) {
      const h = 1e-5;
      const slope = (evaluate(F!, { [v]: x + h, a: 1.3, β: 0.8 }) - evaluate(F!, { [v]: x - h, a: 1.3, β: 0.8 })) / (2 * h);
      expect(slope, src).toBeCloseTo(evaluate(f, { [v]: x, a: 1.3, β: 0.8 }), 4);
    }
  };
  it.each(['sin(x)', 'cos(2x+1)', 'exp(3x)', '1/x', '1/(2x+1)', 'x^(-2)', 'sqrt(x)', '1/(1+x^2)', 'σ(x)', 'σ(βx)', 'tanh(x)', 'ln(x)', 'x·exp(x)', 'x^2·sin(x)', 'x·cos(2x)', '2x·exp(x^2)', 'x·σ(x^2)', '2x/(x^2+1)', 'e^(2x)', '2^x', 'x+sin(x)'])(
    'integrates %s',
    (src) => check(src.replace(/·/g, '*'))
  );
  it('gives up on integrals without an elementary form', () => {
    expect(integrate(parseExpression('x*σ(x)'), 'x')).toBeNull();
  });
});
