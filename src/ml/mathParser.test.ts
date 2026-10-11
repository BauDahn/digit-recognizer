import { describe, expect, it } from 'vitest';
import { evaluate, toLatex, toText } from './mathAst';
import { compileFunction, parseExpression } from './mathParser';
import { simplificationSteps, solve } from './mathSolver';

describe('parser', () => {
  it('respects precedence, parentheses, powers and implicit multiplication', () => {
    expect(evaluate(parseExpression('2+3*4'))).toBe(14);
    expect(evaluate(parseExpression('(2+3)*4'))).toBe(20);
    expect(evaluate(parseExpression('2^3^2'))).toBe(512);
    expect(evaluate(parseExpression('-2^2'))).toBe(-4);
    expect(evaluate(parseExpression('2(3+1)'))).toBe(8);
    expect(evaluate(parseExpression('3x'), 5)).toBe(15);
    expect(evaluate(parseExpression('10-4-3'))).toBe(3);
    expect(evaluate(parseExpression('8/4/2'))).toBe(1);
  });

  it('rejects malformed input', () => {
    expect(() => parseExpression('2+')).toThrow();
    expect(() => parseExpression('(2+3')).toThrow();
    expect(() => parseExpression('2$3')).toThrow();
  });

  it('shows simplification step by step', () => {
    expect(simplificationSteps(parseExpression('24*3+8'))).toEqual(['24 · 3 + 8', '72 + 8', '80']);
  });

  it('compiles functions for plotting without eval', () => {
    const f = compileFunction('x^2-4');
    expect(f(3)).toBe(5);
    expect(f(-2)).toBe(0);
  });
});

describe('solve', () => {
  it('evaluates arithmetic that ends with =', () => {
    const r = solve('24+16=');
    expect(r.kind).toBe('evaluation');
    expect(r.answer).toBe('40');
  });

  it('reads x as multiplication when there is nothing to solve', () => {
    expect(solve('3x4=').answer).toBe('12');
  });

  it('solves linear and quadratic equations', () => {
    expect(solve('2x+3=11').answer).toBe('x = 4');
    expect(solve('x^2=16').answer).toBe('x = -4, x = 4');
    expect(solve('x^2+1=0').answer).toBe('no real solution');
    expect(solve('x^(2)-2x+1=0').answer).toBe('x = 1');
  });

  it('marks expressions in x as graphable', () => {
    const r = solve('x^2-4');
    expect(r.kind).toBe('function');
    expect(compileFunction(r.graphExpr!)(0)).toBe(-4);
  });

  it('reports errors instead of throwing', () => {
    expect(solve('8/(4-4)=').error).toMatch(/Division by zero/);
    expect(solve('1=2=3').kind).toBe('error');
    expect(solve('x^3=8').error).toMatch(/linear and quadratic/);
  });

  it('checks equalities without x', () => {
    expect(solve('2+2=4').answer).toBe('✓');
    expect(solve('2+2=5').answer).toBe('✗');
  });
});

describe('variables, functions and structure', () => {
  const value = (text: string, env: Record<string, number> = {}) => evaluate(parseExpression(text), env);

  it('understands any variable, constants and functions', () => {
    expect(value('2a+b', { a: 3, b: 1 })).toBe(7);
    expect(value('2π')).toBeCloseTo(6.28318, 4);
    expect(value('sin(0)+cos 0')).toBe(1);
    expect(value('sqrt(16)+√9')).toBe(7);
    expect(value('|3-5|')).toBe(2);
    expect(value('4!')).toBe(24);
    expect(value('\\frac(1,4)+\\sqrt(4)')).toBe(2.25);
    expect(value('2·3×4÷6')).toBe(4);
  });

  it('adds up sums and products', () => {
    expect(value('Σ_(i=1)^(4) i^2')).toBe(30);
    expect(value('Σ_(k=0)^(3) 2k+1')).toBe(13); // body is the term: (Σ 2k) + 1
    expect(value('∏_(i=1)^(5) i')).toBe(120);
    expect(value('Σ_(i=1)^(3) (i+1)')).toBe(9);
  });

  it('integrates numerically', () => {
    expect(value('∫_(0)^(2) x^2 dx')).toBeCloseTo(8 / 3, 6);
  });

  it('reads subscripts and exponents with a group', () => {
    expect(toText(parseExpression('x_1^(2)'))).toBe('x_1²');
    expect(toLatex(parseExpression('a_n^(2)+1'))).toBe('a_{n}^{2} + 1');
  });

  it('prints Greek letters and big operators as LaTeX', () => {
    expect(toLatex(parseExpression('Σ_(i=1)^(n) i'))).toBe('\\sum_{i=1}^{n} i');
    expect(toLatex(parseExpression('θ+π'))).toBe('\\theta + \\pi');
  });

  it('rejects malformed structure', () => {
    expect(() => parseExpression('Σ i')).toThrow(/limit/);
    expect(() => parseExpression('∫ x')).toThrow(/dx/);
  });
});

describe('solve: calculus and sums', () => {
  it('evaluates a sum with its steps', () => {
    const r = solve('Σ_(i=1)^(3) i^2=');
    expect(r.answer).toBe('14');
    expect(r.steps).toContain('1² + 2² + 3²');
    expect(r.latex).toBe('\\sum_{i=1}^{3} i^{2} = 14');
  });

  it('differentiates', () => {
    expect(solve('\\frac(d,dx)(x^2+3x)').answer).toBe('2 · x + 3');
    expect(solve('\\frac(∂,∂y)(x^2·y+y^3)').answer).toBe('x² + 3 · y²');
    expect(solve('\\frac(d,dx)(sin(x))').answer).toBe('cos(x)');
    expect(solve('\\frac(d,dx)(x^3)').answer).toBe('3 · x²');
  });

  it('keeps ∂f/∂x as notation when nothing follows it', () => {
    const r = solve('\\frac(∂f,∂x)');
    expect(r.kind).toBe('symbolic');
    expect(r.latex).toBe('\\frac{\\partial f}{\\partial x}');
  });

  it('integrates polynomials', () => {
    expect(solve('∫_(0)^(2) x^2 dx').answer).toBe('2.6666666667');
    expect(solve('∫ 3x^2 dx').answer).toBe('x³ + C');
  });

  it('solves for any single unknown', () => {
    expect(solve('3t-9=0').answer).toBe('t = 3');
  });

  it('only renders what has several unknowns', () => {
    const r = solve('a+b=c');
    expect(r.kind).toBe('symbolic');
    expect(r.latex).toBe('a + b = c');
  });

  it('compares numbers', () => {
    expect(solve('3<5').answer).toBe('✓');
    expect(solve('3≥5').answer).toBe('✗');
  });
});
