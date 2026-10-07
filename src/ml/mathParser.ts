/**
 * Small math engine for the handwriting canvas. No `eval`/`new Function`:
 * a tokenizer + recursive-descent parser build an AST that we can evaluate,
 * simplify step by step, print as text/LaTeX and solve simple equations with.
 *
 * Grammar (lowest to highest precedence):
 *   equation := expr ('=' expr?)?
 *   expr     := term (('+' | '-') term)*
 *   term     := unary (('*' | '/') unary | implicit-multiplication)*
 *   unary    := '-' unary | power
 *   power    := atom ('^' unary)?
 *   atom     := number | 'x' | '(' expr ')'
 */

export type Node =
  | { kind: 'num'; value: number }
  | { kind: 'var' }
  | { kind: 'neg'; arg: Node }
  | { kind: 'bin'; op: '+' | '-' | '*' | '/' | '^'; left: Node; right: Node };

type Token = { type: 'num'; value: number } | { type: 'var' } | { type: 'op'; value: string };

export class MathSyntaxError extends Error {}

export function tokenize(input: string): Token[] {
  const tokens: Token[] = [];
  const src = input.replace(/\s+/g, '').replace(/[×·]/g, '*').replace(/÷/g, '/').replace(/−/g, '-');
  let i = 0;
  while (i < src.length) {
    const ch = src[i];
    if (/[0-9.]/.test(ch)) {
      let j = i;
      while (j < src.length && /[0-9.]/.test(src[j])) j++;
      const text = src.slice(i, j);
      if ((text.match(/\./g) || []).length > 1 || text === '.') {
        throw new MathSyntaxError(`Invalid number: ${text}`);
      }
      tokens.push({ type: 'num', value: parseFloat(text) });
      i = j;
    } else if (ch === 'x' || ch === 'X') {
      tokens.push({ type: 'var' });
      i++;
    } else if ('+-*/^()='.includes(ch)) {
      tokens.push({ type: 'op', value: ch });
      i++;
    } else {
      throw new MathSyntaxError(`Unknown symbol: ${ch}`);
    }
  }
  return tokens;
}

class Parser {
  private pos = 0;
  constructor(private tokens: Token[]) {}

  private peek(): Token | undefined {
    return this.tokens[this.pos];
  }

  private isOp(value: string): boolean {
    const t = this.peek();
    return t?.type === 'op' && t.value === value;
  }

  private expectOp(value: string) {
    if (!this.isOp(value)) throw new MathSyntaxError(`Expected "${value}"`);
    this.pos++;
  }

  atEnd(): boolean {
    return this.pos >= this.tokens.length;
  }

  parseExpr(): Node {
    let node = this.parseTerm();
    while (this.isOp('+') || this.isOp('-')) {
      const op = (this.tokens[this.pos++] as { value: '+' | '-' }).value;
      node = { kind: 'bin', op, left: node, right: this.parseTerm() };
    }
    return node;
  }

  private parseTerm(): Node {
    let node = this.parseUnary();
    for (;;) {
      if (this.isOp('*') || this.isOp('/')) {
        const op = (this.tokens[this.pos++] as { value: '*' | '/' }).value;
        node = { kind: 'bin', op, left: node, right: this.parseUnary() };
      } else if (this.startsAtom()) {
        // Implicit multiplication: 2x, 3(x+1), (x+1)(x-1)
        node = { kind: 'bin', op: '*', left: node, right: this.parsePower() };
      } else {
        return node;
      }
    }
  }

  private startsAtom(): boolean {
    const t = this.peek();
    return !!t && (t.type === 'num' || t.type === 'var' || (t.type === 'op' && t.value === '('));
  }

  private parseUnary(): Node {
    if (this.isOp('-')) {
      this.pos++;
      return { kind: 'neg', arg: this.parseUnary() };
    }
    if (this.isOp('+')) {
      this.pos++;
      return this.parseUnary();
    }
    return this.parsePower();
  }

  private parsePower(): Node {
    const base = this.parseAtom();
    if (this.isOp('^')) {
      this.pos++;
      return { kind: 'bin', op: '^', left: base, right: this.parseUnary() };
    }
    return base;
  }

  private parseAtom(): Node {
    const t = this.peek();
    if (!t) throw new MathSyntaxError('The expression is incomplete');
    if (t.type === 'num') {
      this.pos++;
      return { kind: 'num', value: t.value };
    }
    if (t.type === 'var') {
      this.pos++;
      return { kind: 'var' };
    }
    if (t.value === '(') {
      this.pos++;
      const inner = this.parseExpr();
      this.expectOp(')');
      return inner;
    }
    throw new MathSyntaxError(`Unexpected symbol: ${t.value}`);
  }
}

export function parseExpression(input: string): Node {
  const parser = new Parser(tokenize(input));
  const node = parser.parseExpr();
  if (!parser.atEnd()) throw new MathSyntaxError('Unexpected symbols at the end of the expression');
  return node;
}

export function evaluate(node: Node, x = 0): number {
  switch (node.kind) {
    case 'num':
      return node.value;
    case 'var':
      return x;
    case 'neg':
      return -evaluate(node.arg, x);
    case 'bin': {
      const a = evaluate(node.left, x);
      const b = evaluate(node.right, x);
      switch (node.op) {
        case '+':
          return a + b;
        case '-':
          return a - b;
        case '*':
          return a * b;
        case '/':
          if (b === 0) throw new MathSyntaxError('Division by zero');
          return a / b;
        case '^':
          return Math.pow(a, b);
      }
    }
  }
}

export function containsVariable(node: Node): boolean {
  if (node.kind === 'var') return true;
  if (node.kind === 'neg') return containsVariable(node.arg);
  if (node.kind === 'bin') return containsVariable(node.left) || containsVariable(node.right);
  return false;
}

/** Compile an expression in x into a plain function, for plotting. */
export function compileFunction(input: string): (x: number) => number {
  const ast = parseExpression(input);
  return (x: number) => evaluate(ast, x);
}

const PRECEDENCE = { '+': 1, '-': 1, '*': 2, '/': 2, '^': 4 } as const;

function precedence(node: Node): number {
  if (node.kind === 'bin') return PRECEDENCE[node.op];
  if (node.kind === 'neg' || (node.kind === 'num' && node.value < 0)) return 3;
  return 5;
}

export function formatNumber(value: number): string {
  if (!Number.isFinite(value)) return Number.isNaN(value) ? 'undefined' : value > 0 ? '∞' : '-∞';
  const rounded = Math.round(value * 1e10) / 1e10;
  return String(Object.is(rounded, -0) ? 0 : rounded);
}

export function toText(node: Node): string {
  switch (node.kind) {
    case 'num':
      return formatNumber(node.value);
    case 'var':
      return 'x';
    case 'neg': {
      const inner = toText(node.arg);
      return precedence(node.arg) < 3 ? `-(${inner})` : `-${inner}`;
    }
    case 'bin': {
      const p = PRECEDENCE[node.op];
      // '^' is right-associative, the rest left-associative
      const leftNeedsParens = node.op === '^' ? precedence(node.left) <= p : precedence(node.left) < p;
      const rightNeedsParens =
        precedence(node.right) < p ||
        precedence(node.right) === 3 || // negative operand: 1 - (-1), 2 · (-3)
        (precedence(node.right) === p && (node.op === '-' || node.op === '/'));
      const l = leftNeedsParens ? `(${toText(node.left)})` : toText(node.left);
      const r = rightNeedsParens ? `(${toText(node.right)})` : toText(node.right);
      const opText = node.op === '*' ? ' · ' : node.op === '^' ? '^' : ` ${node.op} `;
      return `${l}${opText}${r}`;
    }
  }
}

export function toLatex(node: Node): string {
  switch (node.kind) {
    case 'num':
      return formatNumber(node.value);
    case 'var':
      return 'x';
    case 'neg':
      return precedence(node.arg) < 3 ? `-\\left(${toLatex(node.arg)}\\right)` : `-${toLatex(node.arg)}`;
    case 'bin': {
      if (node.op === '/') return `\\frac{${toLatex(node.left)}}{${toLatex(node.right)}}`;
      const p = PRECEDENCE[node.op];
      const wrap = (child: Node, needs: boolean) => (needs ? `\\left(${toLatex(child)}\\right)` : toLatex(child));
      if (node.op === '^') {
        return `${wrap(node.left, precedence(node.left) <= p)}^{${toLatex(node.right)}}`;
      }
      const l = wrap(node.left, precedence(node.left) < p);
      const r = wrap(node.right, precedence(node.right) < p || (node.op === '-' && precedence(node.right) === p));
      // Write 2x instead of 2 · x
      const implicit = node.op === '*' && node.left.kind === 'num' && !r.startsWith('\\left') && !/^[0-9]/.test(r);
      const opTex = implicit ? '' : node.op === '*' ? ' \\cdot ' : ` ${node.op} `;
      return `${l}${opTex}${r}`;
    }
  }
}

/**
 * Replace the first (left-most, deepest) operation whose operands are both numbers by its value.
 * Calling it repeatedly shows how an expression is simplified one operation at a time.
 */
function reduceOnce(node: Node): { node: Node; changed: boolean } {
  if (node.kind === 'num' || node.kind === 'var') return { node, changed: false };
  if (node.kind === 'neg') {
    if (node.arg.kind === 'num') return { node: { kind: 'num', value: -node.arg.value }, changed: true };
    const r = reduceOnce(node.arg);
    return { node: { kind: 'neg', arg: r.node }, changed: r.changed };
  }
  const l = reduceOnce(node.left);
  if (l.changed) return { node: { ...node, left: l.node }, changed: true };
  const r = reduceOnce(node.right);
  if (r.changed) return { node: { ...node, right: r.node }, changed: true };
  if (node.left.kind === 'num' && node.right.kind === 'num') {
    return { node: { kind: 'num', value: evaluate(node) }, changed: true };
  }
  return { node, changed: false };
}

export function simplificationSteps(node: Node, maxSteps = 12): string[] {
  const steps = [toText(node)];
  let current = node;
  for (let i = 0; i < 100; i++) {
    const r = reduceOnce(current);
    if (!r.changed) break;
    current = r.node;
    steps.push(toText(current));
  }
  if (steps.length > maxSteps) {
    return [...steps.slice(0, 2), '…', ...steps.slice(-(maxSteps - 3))];
  }
  return steps;
}

export interface SolveResult {
  kind: 'evaluation' | 'equation' | 'function' | 'error';
  input: string;
  latex: string;
  readableText: string;
  answer?: string; // short text to write after '='
  solution?: string;
  steps: string[];
  graphExpr?: string; // expression in x that can be plotted
  error?: string;
}

/**
 * Interpret a recognized line of math:
 *   "12+7="      -> evaluate, answer "19"
 *   "2x+3=11"    -> solve the equation (linear or quadratic)
 *   "x^2-4"      -> a function of x that can be plotted
 *   "3x4="       -> ends with '=' and has nothing to solve, so 'x' means multiplication
 */
export function solve(rawInput: string): SolveResult {
  const input = rawInput.replace(/\s+/g, '');
  const base: SolveResult = { kind: 'error', input, latex: input, readableText: input, steps: [] };
  try {
    const eqCount = (input.match(/=/g) || []).length;
    if (eqCount > 1) throw new MathSyntaxError('More than one "=" sign');

    if (eqCount === 0 || input.endsWith('=')) {
      let exprText = input.replace(/=$/, '');
      let ast = parseExpression(exprText);

      if (containsVariable(ast) && input.endsWith('=')) {
        // "3x4=" : nothing to solve for, so read 'x' as a multiplication sign
        exprText = exprText.replace(/x/gi, '*');
        ast = parseExpression(exprText);
      }

      if (containsVariable(ast)) {
        return {
          ...base,
          kind: 'function',
          latex: `y = ${toLatex(ast)}`,
          readableText: `y = ${toText(ast)}`,
          steps: [`Function of x: y = ${toText(ast)}`, 'It can be plotted as a graph.'],
          graphExpr: toText(ast).replace(/ · /g, '*').replace(/\s/g, ''),
        };
      }

      const value = evaluate(ast);
      const answer = formatNumber(value);
      return {
        ...base,
        kind: 'evaluation',
        latex: `${toLatex(ast)} = ${answer}`,
        readableText: `${toText(ast)} = ${answer}`,
        answer,
        solution: answer,
        steps: simplificationSteps(ast),
      };
    }

    const [lhsText, rhsText] = input.split('=');
    const lhs = parseExpression(lhsText);
    const rhs = parseExpression(rhsText);
    const latex = `${toLatex(lhs)} = ${toLatex(rhs)}`;
    const readable = `${toText(lhs)} = ${toText(rhs)}`;

    if (!containsVariable(lhs) && !containsVariable(rhs)) {
      const a = evaluate(lhs);
      const b = evaluate(rhs);
      const ok = Math.abs(a - b) < 1e-9;
      return {
        ...base,
        kind: 'evaluation',
        latex,
        readableText: readable,
        answer: ok ? '✓' : '✗',
        solution: ok ? 'The equality holds' : `False: ${formatNumber(a)} ≠ ${formatNumber(b)}`,
        steps: [...simplificationSteps(lhs), '=', ...simplificationSteps(rhs)],
      };
    }

    return { ...base, kind: 'equation', latex, readableText: readable, ...solvePolynomial(lhs, rhs) };
  } catch (err) {
    const message = err instanceof Error ? err.message : String(err);
    return { ...base, error: message, steps: [message] };
  }
}

/**
 * Solve f(x) = g(x) when h(x) = f(x) - g(x) is a polynomial of degree <= 2.
 * The coefficients are found by sampling h at x = -1, 0, 1 and verified at other points.
 */
function solvePolynomial(lhs: Node, rhs: Node): Pick<SolveResult, 'answer' | 'solution' | 'steps'> {
  const h = (x: number) => evaluate(lhs, x) - evaluate(rhs, x);
  const hm1 = h(-1);
  const h0 = h(0);
  const h1 = h(1);
  const a = (h1 + hm1) / 2 - h0;
  const b = (h1 - hm1) / 2;
  const c = h0;
  const isPoly = [2, 3, -2.5, 7].every((x) => Math.abs(a * x * x + b * x + c - h(x)) < 1e-6 * (1 + Math.abs(h(x))));
  if (!isPoly) {
    throw new MathSyntaxError('Only linear and quadratic equations can be solved');
  }

  const steps = [`Move everything to one side: ${polyText(a, b, c)} = 0`];
  const eps = 1e-12;

  if (Math.abs(a) < eps) {
    if (Math.abs(b) < eps) {
      const always = Math.abs(c) < eps;
      steps.push(always ? 'True for every x' : 'No solution');
      return { answer: always ? 'x ∈ ℝ' : '∅', solution: always ? 'Infinitely many solutions' : 'No solution', steps };
    }
    const x = -c / b;
    steps.push(`Linear equation: ${formatNumber(b)}·x = ${formatNumber(-c)}`);
    steps.push(`x = ${formatNumber(-c)} / ${formatNumber(b)} = ${formatNumber(x)}`);
    return { answer: `x = ${formatNumber(x)}`, solution: `x = ${formatNumber(x)}`, steps };
  }

  const disc = b * b - 4 * a * c;
  steps.push(`Quadratic equation: a = ${formatNumber(a)}, b = ${formatNumber(b)}, c = ${formatNumber(c)}`);
  steps.push(`Discriminant: b² - 4ac = ${formatNumber(disc)}`);
  if (disc < -eps) {
    steps.push('The discriminant is negative: no real solutions');
    return { answer: 'no real solution', solution: 'No real solutions', steps };
  }
  if (Math.abs(disc) <= eps) {
    const x = -b / (2 * a);
    steps.push(`Double root: x = -b / 2a = ${formatNumber(x)}`);
    return { answer: `x = ${formatNumber(x)}`, solution: `x = ${formatNumber(x)}`, steps };
  }
  const sq = Math.sqrt(disc);
  const x1 = (-b - sq) / (2 * a);
  const x2 = (-b + sq) / (2 * a);
  const [lo, hi] = x1 < x2 ? [x1, x2] : [x2, x1];
  steps.push(`x = (-b ± √disc) / 2a → x₁ = ${formatNumber(lo)}, x₂ = ${formatNumber(hi)}`);
  const text = `x = ${formatNumber(lo)}, x = ${formatNumber(hi)}`;
  return { answer: text, solution: text, steps };
}

function polyText(a: number, b: number, c: number): string {
  const parts: string[] = [];
  const add = (coef: number, suffix: string) => {
    if (Math.abs(coef) < 1e-12) return;
    const sign = coef < 0 ? '-' : '+';
    const abs = Math.abs(coef);
    const num = suffix && Math.abs(abs - 1) < 1e-12 ? '' : formatNumber(abs);
    parts.push(parts.length === 0 ? `${coef < 0 ? '-' : ''}${num}${suffix}` : ` ${sign} ${num}${suffix}`);
  };
  add(a, 'x²');
  add(b, 'x');
  add(c, '');
  return parts.length ? parts.join('') : '0';
}
