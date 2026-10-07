import { describe, expect, it } from 'vitest';
import { compileFunction, evaluate, parseExpression, simplificationSteps, solve } from './mathParser';

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
