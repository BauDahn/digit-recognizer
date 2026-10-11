/**
 * Syntax tree of a handwritten math expression, with everything that can be done to one:
 * evaluate it, print it as text or LaTeX, substitute variables. No `eval` anywhere.
 */
import { symbolLatex } from './symbols';

export class MathSyntaxError extends Error {}

export type BinOp = '+' | '-' | '*' | '/' | '^' | '±';
export type RelOp = '=' | '<' | '>' | '≤' | '≥' | '≠' | '≈';

export type Node =
  | { kind: 'num'; value: number }
  /** A variable. Subscripted ones keep the subscript in the name: "a_n", "x_1". */
  | { kind: 'var'; name: string }
  | { kind: 'const'; name: 'π' | 'e' | '∞'; value: number }
  | { kind: 'neg'; arg: Node }
  | { kind: 'bin'; op: BinOp; left: Node; right: Node }
  | { kind: 'call'; fn: string; arg: Node }
  | { kind: 'fact'; arg: Node }
  /** Summation (Σ) or product (∏) of `body` while `index` goes from `from` to `to`. */
  | { kind: 'bigop'; op: 'Σ' | '∏'; index: string; from: Node; to: Node; body: Node }
  | { kind: 'integral'; variable: string; from?: Node; to?: Node; body: Node }
  /** d/dx or ∂/∂x applied to `body` (or just notation for ∂f/∂x when body is missing). */
  | { kind: 'deriv'; variable: string; partial: boolean; order: number; fn?: string; body?: Node; variables?: string[] }
  | { kind: 'rel'; op: RelOp; left: Node; right: Node };

export const num = (value: number): Node => ({ kind: 'num', value });
/** Placeholder for "nothing yet", the empty side of "12+7=". */
export const EMPTY: Node = { kind: 'var', name: '' };
export const isEmpty = (node: Node): boolean => node.kind === 'var' && node.name === '';
export const variable = (name: string): Node => ({ kind: 'var', name });
export const bin = (op: BinOp, left: Node, right: Node): Node => ({ kind: 'bin', op, left, right });

export const FUNCTIONS: Record<string, (x: number) => number> = {
  sin: Math.sin,
  cos: Math.cos,
  tan: Math.tan,
  cot: (x) => 1 / Math.tan(x),
  sec: (x) => 1 / Math.cos(x),
  csc: (x) => 1 / Math.sin(x),
  arcsin: Math.asin,
  arccos: Math.acos,
  arctan: Math.atan,
  ln: Math.log,
  log: Math.log10,
  exp: Math.exp,
  sqrt: Math.sqrt,
  abs: Math.abs,
  floor: Math.floor,
  ceil: Math.ceil,
  sinh: Math.sinh,
  cosh: Math.cosh,
  tanh: Math.tanh,
  /** σ(x) = 1 / (1 + e^-x), written σ by hand */
  sigmoid: (x) => 1 / (1 + Math.exp(-x)),
  /** swish(x) = x σ(x) (β = 1; write x σ(β x) to choose β) */
  swish: (x) => x / (1 + Math.exp(-x)),
  softplus: (x) => (x > 30 ? x : Math.log1p(Math.exp(x))),
};

export type Env = Record<string, number>;

export function factorial(n: number): number {
  if (!Number.isInteger(n) || n < 0) throw new MathSyntaxError('Factorial needs a whole number ≥ 0');
  if (n > 170) throw new MathSyntaxError('The factorial is too large');
  let out = 1;
  for (let i = 2; i <= n; i++) out *= i;
  return out;
}

/** Evaluate with variable values; a plain number is the value of `x`. */
export function evaluate(node: Node, env: Env | number = {}): number {
  const vars: Env = typeof env === 'number' ? { x: env } : env;
  const ev = (n: Node): number => evaluate(n, vars);
  switch (node.kind) {
    case 'num':
    case 'const':
      return node.value;
    case 'var': {
      const v = vars[node.name];
      if (v === undefined) {
        if (typeof env === 'number') return 0;
        throw new MathSyntaxError(`"${node.name}" has no value`);
      }
      return v;
    }
    case 'neg':
      return -ev(node.arg);
    case 'fact':
      return factorial(ev(node.arg));
    case 'call': {
      const f = FUNCTIONS[node.fn];
      if (!f) throw new MathSyntaxError(`Unknown function ${node.fn}`);
      const arg = ev(node.arg);
      if ((node.fn === 'ln' || node.fn === 'log') && arg <= 0) throw new MathSyntaxError('The logarithm needs a positive number');
      if (node.fn === 'sqrt' && arg < 0) throw new MathSyntaxError('No real square root of a negative number');
      return f(arg);
    }
    case 'bin': {
      const a = ev(node.left);
      const b = ev(node.right);
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
        case '±':
          throw new MathSyntaxError('± has two results, it cannot be computed as one number');
      }
      break;
    }
    case 'bigop': {
      const from = Math.ceil(ev(node.from) - 1e-9);
      const to = Math.floor(ev(node.to) + 1e-9);
      if (to - from > 100000) throw new MathSyntaxError('Too many terms to add up');
      let acc = node.op === 'Σ' ? 0 : 1;
      for (let i = from; i <= to; i++) {
        const term = evaluate(node.body, { ...vars, [node.index]: i });
        acc = node.op === 'Σ' ? acc + term : acc * term;
      }
      return acc;
    }
    case 'integral': {
      if (!node.from || !node.to) throw new MathSyntaxError('An integral needs limits to be computed as a number');
      const a = ev(node.from);
      const b = ev(node.to);
      if (!Number.isFinite(a) || !Number.isFinite(b)) throw new MathSyntaxError('Infinite limits are not supported');
      return simpson((t) => evaluate(node.body, { ...vars, [node.variable]: t }), a, b);
    }
    case 'deriv':
      throw new MathSyntaxError('Derivatives are computed symbolically, not as a number');
    case 'rel':
      throw new MathSyntaxError('This is a comparison, not a number');
  }
}

/** Composite Simpson rule with 2000 intervals. */
export function simpson(f: (x: number) => number, a: number, b: number, n = 2000): number {
  const h = (b - a) / n;
  let sum = f(a) + f(b);
  for (let i = 1; i < n; i++) sum += f(a + i * h) * (i % 2 === 0 ? 2 : 4);
  const result = (sum * h) / 3;
  if (!Number.isFinite(result)) throw new MathSyntaxError('The function is not defined on the whole interval');
  return result;
}

/** Variables that still need a value (the index of a Σ or the variable of an integral is bound). */
export function freeVariables(node: Node, bound: ReadonlySet<string> = new Set()): Set<string> {
  const out = new Set<string>();
  const walk = (n: Node, b: ReadonlySet<string>) => {
    switch (n.kind) {
      case 'var': {
        // a_i with i bound is still a free "a_i": nothing defines the sequence
        if (!b.has(n.name)) out.add(n.name);
        break;
      }
      case 'neg':
      case 'fact':
      case 'call':
        walk(n.arg, b);
        break;
      case 'bin':
      case 'rel':
        walk(n.left, b);
        walk(n.right, b);
        break;
      case 'bigop': {
        walk(n.from, b);
        walk(n.to, b);
        walk(n.body, new Set([...b, n.index]));
        break;
      }
      case 'integral': {
        if (n.from) walk(n.from, b);
        if (n.to) walk(n.to, b);
        walk(n.body, new Set([...b, n.variable]));
        break;
      }
      case 'deriv':
        if (n.body) walk(n.body, b);
        else out.add(n.fn ?? n.variable);
        break;
      default:
        break;
    }
  };
  walk(node, bound);
  return out;
}

export function containsVariable(node: Node): boolean {
  return freeVariables(node).size > 0;
}

/** Replace every free occurrence of the variable `name`. */
export function substitute(node: Node, name: string, value: Node): Node {
  const sub = (n: Node): Node => substitute(n, name, value);
  switch (node.kind) {
    case 'var':
      return node.name === name ? value : node;
    case 'neg':
      return { kind: 'neg', arg: sub(node.arg) };
    case 'fact':
      return { kind: 'fact', arg: sub(node.arg) };
    case 'call':
      return { ...node, arg: sub(node.arg) };
    case 'bin':
    case 'rel':
      return { ...node, left: sub(node.left), right: sub(node.right) } as Node;
    case 'bigop':
      return {
        ...node,
        from: sub(node.from),
        to: sub(node.to),
        body: node.index === name ? node.body : sub(node.body),
      };
    case 'integral':
      return {
        ...node,
        from: node.from && sub(node.from),
        to: node.to && sub(node.to),
        body: node.variable === name ? node.body : sub(node.body),
      };
    case 'deriv':
      return { ...node, body: node.body && sub(node.body) };
    default:
      return node;
  }
}

/** Compile an expression into a plain function of one variable, for plotting. */
export function compileNode(node: Node, name = 'x'): (x: number) => number {
  return (x: number) => evaluate(node, { [name]: x });
}

// ---------------------------------------------------------------- printing

const PRECEDENCE = { '+': 1, '-': 1, '±': 1, '*': 2, '/': 2, '^': 4 } as const;

function precedence(node: Node): number {
  if (node.kind === 'bin') return PRECEDENCE[node.op];
  if (node.kind === 'neg' || (node.kind === 'num' && node.value < 0)) return 3;
  if (node.kind === 'bigop' || node.kind === 'integral' || node.kind === 'deriv') return 2;
  if (node.kind === 'rel') return 0;
  return 5;
}

export function formatNumber(value: number): string {
  if (!Number.isFinite(value)) return Number.isNaN(value) ? 'undefined' : value > 0 ? '∞' : '-∞';
  const rounded = Math.round(value * 1e10) / 1e10;
  return String(Object.is(rounded, -0) ? 0 : rounded);
}

const SUPERSCRIPT_DIGITS = '⁰¹²³⁴⁵⁶⁷⁸⁹';

export function toText(node: Node): string {
  switch (node.kind) {
    case 'num':
      return formatNumber(node.value);
    case 'var':
      return node.name;
    case 'const':
      return node.name;
    case 'neg': {
      const inner = toText(node.arg);
      return precedence(node.arg) < 3 ? `-(${inner})` : `-${inner}`;
    }
    case 'fact': {
      const simple = node.arg.kind === 'num' || node.arg.kind === 'var';
      return simple ? `${toText(node.arg)}!` : `(${toText(node.arg)})!`;
    }
    case 'call':
      return `${node.fn === 'sigmoid' ? 'σ' : node.fn}(${toText(node.arg)})`;
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
      if (node.op === '^' && node.right.kind === 'num' && Number.isInteger(node.right.value)) {
        const exp = node.right.value;
        if (exp >= 0 && exp <= 9) return `${l}${SUPERSCRIPT_DIGITS[exp]}`;
      }
      const opText = node.op === '*' ? ' · ' : node.op === '^' ? '^' : ` ${node.op} `;
      return `${l}${opText}${r}`;
    }
    case 'bigop':
      return `${node.op}[${node.index}=${toText(node.from)}..${toText(node.to)}](${toText(node.body)})`;
    case 'integral': {
      const limits = node.from && node.to ? `[${toText(node.from)}..${toText(node.to)}]` : '';
      return `∫${limits}(${toText(node.body)}) d${node.variable}`;
    }
    case 'deriv': {
      const d = node.partial ? '∂' : 'd';
      const op = node.variables
        ? `${d}${node.order}${node.fn ?? ''}/${node.variables.map((v) => `${d}${v}`).join('')}`
        : `${d}${node.order > 1 ? node.order : ''}${node.fn ?? ''}/${d}${node.variable}${node.order > 1 ? node.order : ''}`;
      return node.body ? `${op}(${toText(node.body)})` : op;
    }
    case 'rel':
      return `${toText(node.left)} ${node.op} ${toText(node.right)}`;
  }
}

const LATEX_FUNCTIONS = new Set(['sin', 'cos', 'tan', 'cot', 'sec', 'csc', 'ln', 'log', 'exp', 'arcsin', 'arccos', 'arctan', 'sinh', 'cosh', 'tanh']);
const LATEX_REL: Record<RelOp, string> = { '=': '=', '<': '<', '>': '>', '≤': '\\leq', '≥': '\\geq', '≠': '\\neq', '≈': '\\approx' };

function variableLatex(name: string): string {
  const [base, sub] = name.split('_');
  const b = symbolLatex(base);
  return sub ? `${b}_{${sub.split('').map(symbolLatex).join('')}}` : b;
}

export function toLatex(node: Node): string {
  const wrap = (child: Node, needs: boolean) => (needs ? `\\left(${toLatex(child)}\\right)` : toLatex(child));
  switch (node.kind) {
    case 'num':
      return formatNumber(node.value);
    case 'var':
      return variableLatex(node.name);
    case 'const':
      return node.name === 'π' ? '\\pi' : node.name === '∞' ? '\\infty' : 'e';
    case 'neg':
      return `-${wrap(node.arg, precedence(node.arg) < 3)}`;
    case 'fact': {
      const simple = node.arg.kind === 'num' || node.arg.kind === 'var';
      return `${wrap(node.arg, !simple)}!`;
    }
    case 'call': {
      if (node.fn === 'sqrt') return `\\sqrt{${toLatex(node.arg)}}`;
      if (node.fn === 'abs') return `\\left|${toLatex(node.arg)}\\right|`;
      if (node.fn === 'floor') return `\\left\\lfloor ${toLatex(node.arg)}\\right\\rfloor`;
      if (node.fn === 'ceil') return `\\left\\lceil ${toLatex(node.arg)}\\right\\rceil`;
      if (node.fn === 'exp') return `e^{${toLatex(node.arg)}}`;
      if (node.fn === 'sigmoid') return `\\sigma\\left(${toLatex(node.arg)}\\right)`;
      const name = LATEX_FUNCTIONS.has(node.fn) ? `\\${node.fn}` : `\\operatorname{${node.fn}}`;
      // names of our own (swish, f, g…) always get parentheses
      const simple = LATEX_FUNCTIONS.has(node.fn) && (node.arg.kind === 'var' || node.arg.kind === 'num' || node.arg.kind === 'const');
      return simple ? `${name} ${toLatex(node.arg)}` : `${name}\\left(${toLatex(node.arg)}\\right)`;
    }
    case 'bin': {
      if (node.op === '/') return `\\frac{${toLatex(node.left)}}{${toLatex(node.right)}}`;
      const p = PRECEDENCE[node.op];
      if (node.op === '^') {
        const base = node.left;
        const baseNeedsParens = precedence(base) <= p || (base.kind === 'call' && base.fn !== 'sqrt');
        return `${wrap(base, baseNeedsParens)}^{${toLatex(node.right)}}`;
      }
      const l = wrap(node.left, precedence(node.left) < p);
      const r = wrap(node.right, precedence(node.right) < p || (node.op === '-' && precedence(node.right) === p));
      // Write 2x instead of 2 · x
      const rIsSymbol = /^[A-Za-z\\]/.test(r) && !r.startsWith('\\left');
      const implicit = node.op === '*' && node.left.kind !== 'bin' && !r.startsWith('\\left') && !/^[0-9]/.test(r) && rIsSymbol;
      const opTex = implicit ? '' : node.op === '*' ? ' \\cdot ' : node.op === '±' ? ' \\pm ' : ` ${node.op} `;
      // \beta x, not \betax
      const glue = opTex === '' && /\\[A-Za-z]+$/.test(l) && /^[A-Za-z]/.test(r) ? ' ' : '';
      return `${l}${opTex}${glue}${r}`;
    }
    case 'bigop': {
      const body = wrap(node.body, precedence(node.body) < 2);
      const symbol = node.op === 'Σ' ? '\\sum' : '\\prod';
      return `${symbol}_{${variableLatex(node.index)}=${toLatex(node.from)}}^{${toLatex(node.to)}} ${body}`;
    }
    case 'integral': {
      const limits = node.from && node.to ? `_{${toLatex(node.from)}}^{${toLatex(node.to)}}` : '';
      return `\\int${limits} ${toLatex(node.body)}\\,d${variableLatex(node.variable)}`;
    }
    case 'deriv': {
      const d = node.partial ? '\\partial' : 'd';
      const sup = node.order > 1 ? `^{${node.order}}` : '';
      const fn = node.fn ? ` ${variableLatex(node.fn)}` : '';
      const op = node.variables
        ? `\\frac{${d}${sup}${fn}}{${node.variables.map((v) => `${d} ${variableLatex(v)}`).join('\\,')}}`
        : `\\frac{${d}${sup}${fn}}{${d} ${variableLatex(node.variable)}${sup}}`;
      return node.body ? `${op}\\left(${toLatex(node.body)}\\right)` : op;
    }
    case 'rel':
      return `${toLatex(node.left)} ${LATEX_REL[node.op]} ${toLatex(node.right)}`;
  }
}
