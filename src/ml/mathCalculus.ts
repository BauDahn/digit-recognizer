/** Symbolic calculus on the syntax tree: derivatives, a light simplifier and polynomial antiderivatives. */
import { bin, evaluate, freeVariables, MathSyntaxError, Node, num, substitute as substituteVar, variable } from './mathAst';

const ZERO = num(0);
const ONE = num(1);

const dependsOn = (node: Node, v: string): boolean => freeVariables(node).has(v);

/** Derivative of `node` with respect to `v` (every other variable is a constant). */
export function differentiate(node: Node, v: string, order = 1): Node {
  let current = node;
  for (let i = 0; i < order; i++) current = simplify(derivative(current, v));
  return current;
}

/** ∂²f/∂x∂y …: one derivative after the other. */
export function differentiateAll(node: Node, variables: string[]): Node {
  return variables.reduce((current, v) => differentiate(current, v), node);
}

const sigmoidOf = (u: Node): Node => ({ kind: 'call', fn: 'sigmoid', arg: u });
const callOf = (fn: string, u: Node): Node => ({ kind: 'call', fn, arg: u });

function derivative(node: Node, v: string): Node {
  if (!dependsOn(node, v)) return ZERO;
  const d = (n: Node) => derivative(n, v);
  switch (node.kind) {
    case 'var':
      return ONE; // dependsOn guaranteed the name is v
    case 'neg':
      return { kind: 'neg', arg: d(node.arg) };
    case 'bin': {
      const { left: a, right: b } = node;
      switch (node.op) {
        case '+':
        case '-':
          return bin(node.op, d(a), d(b));
        case '*':
          return bin('+', bin('*', d(a), b), bin('*', a, d(b)));
        case '/':
          return bin('/', bin('-', bin('*', d(a), b), bin('*', a, d(b))), bin('^', b, num(2)));
        case '^': {
          if (!dependsOn(b, v)) {
            // power rule + chain rule: n·u^(n-1)·u'
            return bin('*', bin('*', b, bin('^', a, bin('-', b, ONE))), d(a));
          }
          if (!dependsOn(a, v)) {
            // a^u · ln(a) · u'
            return bin('*', bin('*', node, { kind: 'call', fn: 'ln', arg: a }), d(b));
          }
          // u^w = e^(w ln u)
          return bin(
            '*',
            node,
            bin('+', bin('*', d(b), { kind: 'call', fn: 'ln', arg: a }), bin('/', bin('*', b, d(a)), a))
          );
        }
        case '±':
          throw new MathSyntaxError('Cannot differentiate ±');
      }
      break;
    }
    case 'call': {
      const u = node.arg;
      const du = d(u);
      let outer: Node;
      switch (node.fn) {
        case 'sin':
          outer = { kind: 'call', fn: 'cos', arg: u };
          break;
        case 'cos':
          outer = { kind: 'neg', arg: { kind: 'call', fn: 'sin', arg: u } };
          break;
        case 'tan':
          outer = bin('/', ONE, bin('^', { kind: 'call', fn: 'cos', arg: u }, num(2)));
          break;
        case 'cot':
          outer = { kind: 'neg', arg: bin('/', ONE, bin('^', callOf('sin', u), num(2))) };
          break;
        case 'sec':
          outer = bin('*', node, callOf('tan', u));
          break;
        case 'csc':
          outer = { kind: 'neg', arg: bin('*', node, callOf('cot', u)) };
          break;
        case 'sinh':
          outer = callOf('cosh', u);
          break;
        case 'cosh':
          outer = callOf('sinh', u);
          break;
        case 'tanh':
          outer = bin('-', ONE, bin('^', node, num(2)));
          break;
        case 'sigmoid':
          // σ'(u) = σ(u)(1 - σ(u))
          outer = bin('*', node, bin('-', ONE, node));
          break;
        case 'softplus':
          outer = sigmoidOf(u);
          break;
        case 'swish':
          // (u σ(u))' = σ(u) + u σ(u)(1 - σ(u))
          outer = bin('+', sigmoidOf(u), bin('*', u, bin('*', sigmoidOf(u), bin('-', ONE, sigmoidOf(u)))));
          break;
        case 'abs':
          outer = bin('/', u, node);
          break;
        case 'exp':
          outer = node;
          break;
        case 'ln':
          outer = bin('/', ONE, u);
          break;
        case 'log':
          outer = bin('/', ONE, bin('*', u, { kind: 'call', fn: 'ln', arg: num(10) }));
          break;
        case 'sqrt':
          outer = bin('/', ONE, bin('*', num(2), node));
          break;
        case 'arcsin':
          outer = bin('/', ONE, { kind: 'call', fn: 'sqrt', arg: bin('-', ONE, bin('^', u, num(2))) });
          break;
        case 'arccos':
          outer = { kind: 'neg', arg: bin('/', ONE, { kind: 'call', fn: 'sqrt', arg: bin('-', ONE, bin('^', u, num(2))) }) };
          break;
        case 'arctan':
          outer = bin('/', ONE, bin('+', ONE, bin('^', u, num(2))));
          break;
        default:
          throw new MathSyntaxError(`Cannot differentiate ${node.fn}`);
      }
      return bin('*', outer, du);
    }
    default:
      throw new MathSyntaxError('Cannot differentiate this expression');
  }
  throw new MathSyntaxError('Cannot differentiate this expression');
}

const sameNode = (a: Node, b: Node): boolean => JSON.stringify(a) === JSON.stringify(b);
const isNum = (n: Node): n is { kind: 'num'; value: number } => n.kind === 'num';
/** A number with exactly this value (not a type guard, so the other branch keeps its type). */
const isVal = (n: Node, value: number): boolean => n.kind === 'num' && n.value === value;

/** Constant folding and the obvious identities (x·1, x+0, x^1 ...). Not a full CAS. */
export function simplify(node: Node): Node {
  switch (node.kind) {
    case 'neg': {
      const arg = simplify(node.arg);
      if (isNum(arg)) return num(-arg.value);
      if (arg.kind === 'neg') return arg.arg;
      return { kind: 'neg', arg };
    }
    case 'call':
      return { ...node, arg: simplify(node.arg) };
    case 'bin': {
      const a = simplify(node.left);
      const b = simplify(node.right);
      switch (node.op) {
        case '+':
          if (isVal(a, 0)) return b;
          if (isVal(b, 0)) return a;
          if (isNum(a) && isNum(b)) return num(a.value + b.value);
          if (b.kind === 'neg') return simplify(bin('-', a, b.arg));
          return bin('+', a, b);
        case '-':
          if (isVal(b, 0)) return a;
          if (isVal(a, 0)) return simplify({ kind: 'neg', arg: b });
          if (isNum(a) && isNum(b)) return num(a.value - b.value);
          if (sameNode(a, b)) return ZERO;
          return bin('-', a, b);
        case '*': {
          if (isVal(a, 0) || isVal(b, 0)) return ZERO;
          if (isVal(a, 1)) return b;
          if (isVal(b, 1)) return a;
          if (isNum(a) && isNum(b)) return num(a.value * b.value);
          if (a.kind === 'neg') return simplify({ kind: 'neg', arg: bin('*', a.arg, b) });
          if (b.kind === 'neg') return simplify({ kind: 'neg', arg: bin('*', a, b.arg) });
          // one numeric coefficient, in front: x·2·y·3 → 6·x·y
          const factors = [...flatProduct(a), ...flatProduct(b)];
          const coefficient = factors.reduce((acc, f) => (isNum(f) ? acc * f.value : acc), 1);
          const others = factors.filter((f) => !isNum(f));
          if (coefficient === 0) return ZERO;
          return productOf(coefficient !== 1 || others.length === 0 ? [num(coefficient), ...others] : others);
        }
        case '/':
          if (isVal(a, 0)) return ZERO;
          if (isVal(b, 1)) return a;
          if (sameNode(a, b)) return ONE;
          if (isNum(a) && isNum(b) && b.value !== 0) {
            const q = a.value / b.value;
            if (Number.isInteger(q)) return num(q);
          }
          return bin('/', a, b);
        case '^':
          if (isVal(b, 0)) return ONE;
          if (isVal(b, 1)) return a;
          if (isNum(a) && isNum(b) && Number.isInteger(b.value) && b.value >= 0) return num(Math.pow(a.value, b.value));
          return bin('^', a, b);
        default:
          return bin(node.op, a, b);
      }
    }
    default:
      return node;
  }
}

/** Polynomial in `v` as exponent → coefficient, or null if `node` is not one. */
function polynomial(node: Node, v: string): Map<number, Node> | null {
  if (!dependsOn(node, v)) return new Map([[0, node]]);
  switch (node.kind) {
    case 'var':
      return new Map([[1, ONE]]);
    case 'neg': {
      const p = polynomial(node.arg, v);
      return p && new Map([...p].map(([e, c]) => [e, simplify({ kind: 'neg', arg: c })]));
    }
    case 'bin': {
      if (node.op === '+' || node.op === '-') {
        const l = polynomial(node.left, v);
        const r = polynomial(node.right, v);
        if (!l || !r) return null;
        const out = new Map(l);
        for (const [e, c] of r) {
          const existing = out.get(e);
          const term = node.op === '-' ? simplify({ kind: 'neg', arg: c }) : c;
          out.set(e, existing ? simplify(bin('+', existing, term)) : term);
        }
        return out;
      }
      if (node.op === '*') {
        const l = polynomial(node.left, v);
        const r = polynomial(node.right, v);
        if (!l || !r) return null;
        const out = new Map<number, Node>();
        for (const [e1, c1] of l) {
          for (const [e2, c2] of r) {
            const e = e1 + e2;
            const term = simplify(bin('*', c1, c2));
            out.set(e, out.has(e) ? simplify(bin('+', out.get(e)!, term)) : term);
          }
        }
        return out;
      }
      if (node.op === '/' && !dependsOn(node.right, v)) {
        const l = polynomial(node.left, v);
        return l && new Map([...l].map(([e, c]) => [e, simplify(bin('/', c, node.right))]));
      }
      if (node.op === '^' && isNum(node.right) && Number.isInteger(node.right.value) && node.right.value >= 0) {
        let acc: Map<number, Node> | null = new Map([[0, ONE]]);
        for (let i = 0; i < node.right.value && acc; i++) {
          acc = polynomial(bin('*', nodeFromPoly(acc, v), node.left), v);
        }
        return acc;
      }
      return null;
    }
    default:
      return null;
  }
}

function nodeFromPoly(poly: Map<number, Node>, v: string): Node {
  let out: Node | null = null;
  for (const [e, c] of [...poly].sort((a, b) => b[0] - a[0])) {
    const power: Node = e === 0 ? ONE : e === 1 ? variable(v) : bin('^', variable(v), num(e));
    const term = simplify(bin('*', c, power));
    if (isVal(term, 0)) continue;
    out = out ? simplify(bin('+', out, term)) : term;
  }
  return out ?? ZERO;
}

/** Antiderivative of a polynomial (without the constant of integration), or null. */
function integratePolynomial(node: Node, v: string): Node | null {
  const poly = polynomial(node, v);
  if (!poly) return null;
  const anti = new Map<number, Node>();
  for (const [e, c] of poly) anti.set(e + 1, simplify(bin('/', c, num(e + 1))));
  return nodeFromPoly(anti, v);
}

/**
 * Antiderivative (without the constant of integration), or null when none is found.
 * Polynomials, sums and constant multiples, powers and elementary functions of a linear argument,
 * polynomial × e^x / sin / cos by parts, and u-substitution f(g(x))·g'(x).
 */
export function integrate(node: Node, v: string): Node | null {
  const poly = integratePolynomial(node, v);
  if (poly) return poly;
  try {
    const result = antiderivative(node, v, 0);
    return result && simplify(result);
  } catch {
    return null;
  }
}

/** a·v + b with a and b free of v. */
function linearIn(node: Node, v: string): { a: Node; b: Node } | null {
  if (!dependsOn(node, v)) return null;
  let a: Node;
  try {
    a = simplify(derivative(node, v));
  } catch {
    return null;
  }
  if (dependsOn(a, v) || isVal(a, 0)) return null;
  return { a, b: simplify(substituteVar(node, v, ZERO)) };
}

const lnAbs = (u: Node): Node => callOf('ln', callOf('abs', u));

function flatProduct(node: Node): Node[] {
  return node.kind === 'bin' && node.op === '*' ? [...flatProduct(node.left), ...flatProduct(node.right)] : [node];
}
const productOf = (factors: Node[]): Node => factors.reduce((acc, f) => bin('*', acc, f));

function antiderivative(node: Node, v: string, depth: number): Node | null {
  if (depth > 6) return null;
  if (!dependsOn(node, v)) return bin('*', node, variable(v));
  const again = (n: Node) => antiderivative(n, v, depth + 1);

  switch (node.kind) {
    case 'neg': {
      const inner = again(node.arg);
      return inner && { kind: 'neg', arg: inner };
    }
    case 'var':
      return bin('/', bin('^', node, num(2)), num(2));
    case 'bin': {
      const { left, right } = node;
      switch (node.op) {
        case '+':
        case '-': {
          const l = again(left);
          const r = again(right);
          return l && r && bin(node.op, l, r);
        }
        case '*': {
          if (!dependsOn(left, v)) {
            const r = again(right);
            return r && bin('*', left, r);
          }
          if (!dependsOn(right, v)) {
            const l = again(left);
            return l && bin('*', right, l);
          }
          return byParts(node, v, depth) ?? substitution(node, v, depth);
        }
        case '/': {
          if (!dependsOn(right, v)) {
            const l = again(left);
            return l && bin('/', l, right);
          }
          if (!dependsOn(left, v)) {
            const lin = linearIn(right, v);
            if (lin) return bin('*', bin('/', left, lin.a), lnAbs(right));
            // c / (k + L²) → arctan
            const atan = arctanForm(left, right, v);
            if (atan) return atan;
          }
          return substitution(node, v, depth);
        }
        case '^': {
          const { left: base, right: exponent } = node;
          if (!dependsOn(exponent, v)) {
            const lin = linearIn(base, v);
            if (!lin) return null;
            if (exponent.kind === 'num' && exponent.value === -1) return bin('/', lnAbs(base), lin.a);
            const n1 = simplify(bin('+', exponent, ONE));
            return bin('/', bin('^', base, n1), bin('*', n1, lin.a));
          }
          if (!dependsOn(base, v)) {
            const lin = linearIn(exponent, v);
            if (!lin) return null;
            if (base.kind === 'const' && base.name === 'e') return bin('/', node, lin.a);
            return bin('/', node, bin('*', callOf('ln', base), lin.a));
          }
          return null;
        }
        default:
          return null;
      }
    }
    case 'call': {
      const lin = linearIn(node.arg, v);
      if (!lin) return null;
      const u = node.arg;
      let anti: Node;
      switch (node.fn) {
        case 'sin':
          anti = { kind: 'neg', arg: callOf('cos', u) };
          break;
        case 'cos':
          anti = callOf('sin', u);
          break;
        case 'tan':
          anti = { kind: 'neg', arg: lnAbs(callOf('cos', u)) };
          break;
        case 'exp':
          anti = node;
          break;
        case 'sinh':
          anti = callOf('cosh', u);
          break;
        case 'cosh':
          anti = callOf('sinh', u);
          break;
        case 'tanh':
          anti = callOf('ln', callOf('cosh', u));
          break;
        case 'sigmoid':
          anti = callOf('softplus', u);
          break;
        case 'ln':
          anti = bin('-', bin('*', u, node), u);
          break;
        case 'sqrt':
          anti = bin('*', bin('/', num(2), num(3)), bin('^', u, bin('/', num(3), num(2))));
          break;
        case 'arctan':
          anti = bin('-', bin('*', u, node), bin('/', callOf('ln', bin('+', ONE, bin('^', u, num(2)))), num(2)));
          break;
        default:
          return null;
      }
      return bin('/', anti, lin.a);
    }
    default:
      return null;
  }
}

/** c / (k + L²) with a positive number k and L linear in v. */
function arctanForm(numerator: Node, denominator: Node, v: string): Node | null {
  if (denominator.kind !== 'bin' || denominator.op !== '+') return null;
  const [k, sq] = isNum(denominator.left) ? [denominator.left, denominator.right] : [denominator.right, denominator.left];
  if (!isNum(k) || k.value <= 0 || sq.kind !== 'bin' || sq.op !== '^' || !isVal(sq.right, 2)) return null;
  const lin = linearIn(sq.left, v);
  if (!lin) return null;
  const root = Math.sqrt(k.value);
  const scaled = root === 1 ? sq.left : bin('/', sq.left, num(root));
  return bin('/', bin('*', numerator, callOf('arctan', scaled)), bin('*', num(root), lin.a));
}

/** Tabular integration by parts for polynomial × e^(ax+b), sin(ax+b) or cos(ax+b). */
function byParts(node: Node, v: string, depth: number): Node | null {
  if (node.kind !== 'bin' || node.op !== '*') return null;
  for (const [p, e] of [
    [node.left, node.right],
    [node.right, node.left],
  ]) {
    if (!polynomial(p, v) || e.kind !== 'call' || !['exp', 'sin', 'cos'].includes(e.fn)) continue;
    let current = p;
    let integral: Node | null = e;
    let sign = 1;
    let total: Node | null = null;
    for (let step = 0; step < 8 && integral; step++) {
      integral = antiderivative(integral, v, depth + 1);
      if (!integral) return null;
      const term: Node = bin('*', current, integral);
      const signed: Node = sign < 0 ? { kind: 'neg', arg: term } : term;
      total = total ? bin('+', total, signed) : signed;
      current = simplify(derivative(current, v));
      if (isVal(current, 0)) return total;
      sign = -sign;
    }
  }
  return null;
}

/**
 * ∫ F(g(x))·g'(x) dx = G(g(x)) where G' = F. The leftover factor must be a constant multiple of g'
 * (checked at sample points), so that e.g. ∫ x·σ(x²) dx = ½ softplus(x²).
 */
function substitution(node: Node, v: string, depth: number): Node | null {
  const parts = node.kind === 'bin' && node.op === '/' ? [...flatProduct(node.left), bin('^', node.right, num(-1))] : flatProduct(node);
  const U = '\u0000';
  for (let i = 0; i < parts.length; i++) {
    const factor = parts[i];
    let inner: Node;
    let outer: Node;
    if (factor.kind === 'call') {
      inner = factor.arg;
      outer = { ...factor, arg: variable(U) };
    } else if (factor.kind === 'bin' && factor.op === '^' && !dependsOn(factor.right, v)) {
      inner = factor.left;
      outer = bin('^', variable(U), factor.right);
    } else continue;
    if (inner.kind === 'var' || linearIn(inner, v)) continue;

    const rest = parts.length === 1 ? ONE : productOf(parts.filter((_, j) => j !== i));
    const du = simplify(derivative(inner, v));
    const k = constantRatio(rest, du, v);
    if (k === null) continue;
    const anti = antiderivative(outer, U, depth + 1);
    if (!anti) continue;
    return bin('*', num(k), substituteVar(anti, U, inner));
  }
  return null;
}

/** rest / du as a number if it does not change with v nor with any other variable, else null. */
function constantRatio(rest: Node, du: Node, v: string): number | null {
  const names = [...new Set([...freeVariables(rest), ...freeVariables(du)])];
  const samples = [
    [0.31, 0.6],
    [0.93, 1.3],
    [1.7, 0.9],
    [0.52, 1.8],
  ];
  let first: number | null = null;
  for (const [x, other] of samples) {
    const env: Record<string, number> = {};
    names.forEach((name, i) => (env[name] = name === v ? x : other + 0.37 * i));
    let ratio: number;
    try {
      ratio = evaluate(rest, env) / evaluate(du, env);
    } catch {
      return null;
    }
    if (!Number.isFinite(ratio)) return null;
    if (first === null) first = ratio;
    else if (Math.abs(ratio - first) > 1e-7 * (1 + Math.abs(first))) return null;
  }
  return first;
}
