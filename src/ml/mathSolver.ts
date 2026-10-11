/**
 * Interpret a recognized line of math and, when possible, compute it:
 *
 *   "12+7="            evaluate, answer "19"
 *   "2x+3=11"          solve the equation (linear or quadratic)
 *   "x^2-4"            a function of x that can be plotted
 *   "Σ_(i=1)^(4) i^2=" add the terms up
 *   "d/dx(x^2+3x)"     differentiate symbolically
 *   "∫_(0)^(2) x^2 dx" integrate
 *   "3x4="             ends with '=' and has nothing to solve, so 'x' means multiplication
 *
 * Anything else that parses is still turned into LaTeX ("symbolic").
 */
import {
  bin,
  containsVariable,
  evaluate,
  formatNumber,
  freeVariables,
  isEmpty,
  MathSyntaxError,
  Node,
  num,
  RelOp,
  substitute,
  toLatex,
  toText,
  variable,
} from './mathAst';
import { differentiate, differentiateAll, integrate, simplify } from './mathCalculus';
import { collectDefinitions, definedFunctionName, Definitions, inlineDefinitions, NO_DEFINITIONS } from './mathContext';
import { parseRelation } from './mathParser';

export { collectDefinitions };
export type { Definitions };

export interface SolveResult {
  kind: 'evaluation' | 'equation' | 'function' | 'symbolic' | 'error';
  input: string;
  latex: string;
  readableText: string;
  answer?: string; // short text to write after '='
  solution?: string;
  steps: string[];
  graphExpr?: string; // expression in x that can be plotted
  error?: string;
}

// ---------------------------------------------------------------- step by step

/**
 * Replace the first (left-most, deepest) operation whose operands are all numbers by its value.
 * Calling it repeatedly shows how an expression is simplified one operation at a time.
 */
function reduceOnce(node: Node): { node: Node; changed: boolean } {
  switch (node.kind) {
    case 'num':
    case 'var':
    case 'const':
    case 'rel':
    case 'deriv':
    case 'integral':
      return { node, changed: false };
    case 'neg': {
      if (node.arg.kind === 'num') return { node: num(-node.arg.value), changed: true };
      const r = reduceOnce(node.arg);
      return { node: { kind: 'neg', arg: r.node }, changed: r.changed };
    }
    case 'fact':
    case 'call': {
      if (node.arg.kind === 'num') return { node: num(evaluate(node)), changed: true };
      const r = reduceOnce(node.arg);
      return { node: { ...node, arg: r.node } as Node, changed: r.changed };
    }
    case 'bigop': {
      const l = reduceOnce(node.from);
      if (l.changed) return { node: { ...node, from: l.node }, changed: true };
      const u = reduceOnce(node.to);
      if (u.changed) return { node: { ...node, to: u.node }, changed: true };
      if (node.from.kind === 'num' && node.to.kind === 'num') return { node: expandBigOp(node), changed: true };
      return { node, changed: false };
    }
    case 'bin': {
      const l = reduceOnce(node.left);
      if (l.changed) return { node: { ...node, left: l.node }, changed: true };
      const r = reduceOnce(node.right);
      if (r.changed) return { node: { ...node, right: r.node }, changed: true };
      if (node.left.kind === 'num' && node.right.kind === 'num' && node.op !== '±') {
        return { node: num(evaluate(node)), changed: true };
      }
      return { node, changed: false };
    }
  }
}

/** Σ_(i=1)^(3) i² → 1² + 2² + 3² (or just the value when there are many terms). */
function expandBigOp(node: Extract<Node, { kind: 'bigop' }>): Node {
  const from = Math.ceil((node.from as { value: number }).value - 1e-9);
  const to = Math.floor((node.to as { value: number }).value + 1e-9);
  if (to < from) return num(node.op === 'Σ' ? 0 : 1);
  if (to - from + 1 > 6) return num(evaluate(node));
  let acc: Node | null = null;
  for (let i = from; i <= to; i++) {
    const term = substitute(node.body, node.index, num(i));
    acc = acc ? bin(node.op === 'Σ' ? '+' : '*', acc, term) : term;
  }
  return acc!;
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
  if (current.kind !== 'num') {
    // constants such as π or e: finish with the decimal value
    const value = formatNumber(evaluate(current));
    if (value !== steps[steps.length - 1]) steps.push(`≈ ${value}`);
  }
  if (steps.length > maxSteps) {
    return [...steps.slice(0, 2), '…', ...steps.slice(-(maxSteps - 3))];
  }
  return steps;
}

// ---------------------------------------------------------------- solve

const EQUATION_LATEX: Record<RelOp, string> = { '=': '=', '<': '<', '>': '>', '≤': '\\leq', '≥': '\\geq', '≠': '\\neq', '≈': '\\approx' };

/** Solve a line. `defs` are the definitions of the other lines of the page (see collectDefinitions). */
export function solve(rawInput: string, defs: Definitions = NO_DEFINITIONS): SolveResult {
  const input = rawInput.replace(/\s+/g, '').replace(/\?+$/, '');
  const base: SolveResult = { kind: 'error', input, latex: input, readableText: input, steps: [] };
  try {
    const own = definedFunctionName(input);
    const userFunctions = new Set([...defs.functions.keys(), ...(own ? [own] : [])]);
    let ast = parseRelation(input, userFunctions);

    // "3x4=": nothing to solve for, so 'x' means multiplication
    if (ast.kind === 'rel' && isEmpty(ast.right)) {
      const vars = freeVariables(ast.left);
      if (vars.size === 1 && vars.has('x') && /x/.test(input)) {
        try {
          ast = parseRelation(input.replace(/x/g, '*'), userFunctions);
        } catch {
          // keep reading x as a variable
        }
      }
    }

    const definition = describeDefinition(base, ast, defs);
    if (definition) return definition;
    ast = inlineDefinitions(ast, defs);

    if (ast.kind === 'rel') {
      if (isEmpty(ast.right)) return evaluateExpression(base, ast.left, true, defs);
      return solveRelation(base, ast.op, ast.left, ast.right);
    }
    return evaluateExpression(base, ast, false, defs);
  } catch (err) {
    const message = err instanceof Error ? err.message : String(err);
    return { ...base, error: message, steps: [message] };
  }
}

/** "swish(x) = xσ(βx)" and "y = swish(x)": they define something, and can be plotted when they have one unknown. */
function describeDefinition(base: SolveResult, ast: Node, defs: Definitions): SolveResult | null {
  if (ast.kind !== 'rel' || ast.op !== '=' || isEmpty(ast.right)) return null;
  const { left, right } = ast;
  let param: string | null = null;
  if (left.kind === 'call' && left.arg.kind === 'var' && defs.functions.get(left.fn)) param = left.arg.name;
  else if (left.kind === 'var' && left.name && defs.variables.has(left.name)) param = '';
  else if (left.kind === 'call' && left.arg.kind === 'var' && definedFunctionName(base.input) === left.fn) param = left.arg.name;
  if (param === null) return null;

  const body = inlineDefinitions(right, defs);
  const unknowns = [...freeVariables(body)];
  const latex = `${toLatex(left)} = ${toLatex(right)}`;
  const readableText = `${toText(left)} = ${toText(right)}`;
  const name = toText(left);
  const lines = [`Definition: ${readableText}`];
  if (body !== right && toText(body) !== toText(right)) lines.push(`With the other definitions: ${name} = ${toText(body)}`);

  const plotted = param ? unknowns.length === 1 && unknowns[0] === param : unknowns.length === 1;
  if (plotted) {
    const [v] = unknowns;
    const inX = v === 'x' ? body : substitute(body, v, variable('x'));
    return {
      ...base,
      kind: 'function',
      latex,
      readableText,
      steps: [...lines, 'It can be plotted as a graph.'],
      graphExpr: toText(inX).replace(/ · /g, '*').replace(/\s/g, ''),
    };
  }
  const constants = unknowns.filter((u) => u !== param);
  if (param && constants.length > 0) {
    lines.push(`${constants.join(', ')} ${constants.length > 1 ? 'are' : 'is'} treated as a constant when differentiating or integrating.`);
  }
  lines.push(`Write ∂${param ? '' : name}/∂${param || 'x'} on another line to differentiate it.`);
  return { ...base, kind: 'symbolic', latex, readableText, steps: lines };
}

function evaluateExpression(base: SolveResult, ast: Node, trailingEquals: boolean, defs: Definitions): SolveResult {
  // d/dx and ∫ are computed symbolically
  const calculus = solveCalculus(base, ast, defs);
  if (calculus) return calculus;

  const vars = [...freeVariables(ast)];
  if (vars.length === 0) {
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

  if (vars.length === 1 && !trailingEquals) {
    // A function of one variable: it can be plotted. It is renamed to x for the plot.
    const [v] = vars;
    const inX = v === 'x' ? ast : substitute(ast, v, variable('x'));
    return {
      ...base,
      kind: 'function',
      latex: `y = ${toLatex(ast)}`,
      readableText: `y = ${toText(ast)}`,
      steps: [`Function of ${v}: y = ${toText(ast)}`, 'It can be plotted as a graph.'],
      graphExpr: toText(inX).replace(/ · /g, '*').replace(/\s/g, ''),
    };
  }

  return symbolic(base, toLatex(ast), toText(ast), `It has unknowns (${vars.join(', ')}), so it can only be written, not computed.`);
}

function symbolic(base: SolveResult, latex: string, readableText: string, note: string): SolveResult {
  return { ...base, kind: 'symbolic', latex, readableText, steps: [note] };
}

function solveCalculus(base: SolveResult, ast: Node, defs: Definitions): SolveResult | null {
  if (ast.kind === 'deriv') {
    const named = ast.fn && !ast.body ? namedExpression(ast.fn, defs) : null;
    const body = ast.body ?? named?.body;
    if (!body) {
      return symbolic(
        base,
        toLatex(ast),
        toText(ast),
        ast.fn
          ? `"${ast.fn}" is not defined on this page. Write it on another line, like ${ast.fn} = x² + 3x, and this line is solved too.`
          : 'This is derivative notation: write what to differentiate after it, like d/dx (x² + 3x).'
      );
    }
    const variables = ast.variables ?? Array(ast.order).fill(ast.variable);
    const result = simplify(ast.variables ? differentiateAll(body, variables) : differentiate(body, ast.variable, ast.order));
    const text = toText(result);
    const constants = [...freeVariables(body)].filter((name) => !variables.includes(name));
    const steps: string[] = [];
    if (named) steps.push(`${ast.fn} = ${toText(named.defined)}`);
    if (named && toText(named.defined) !== toText(body)) steps.push(`  = ${toText(body)}`);
    steps.push(
      `${ast.partial ? '∂' : 'd'}/${ast.partial ? '∂' : 'd'}${variables.join(ast.partial ? '∂' : 'd')} of ${toText(body)}` +
        (constants.length > 0 ? `, with ${constants.join(', ')} constant` : '')
    );
    steps.push(...rulesUsed(body, variables[0]));
    steps.push(`= ${text}`);
    return {
      ...base,
      kind: 'evaluation',
      latex: `${toLatex(ast)} = ${toLatex(result)}`,
      readableText: `${toText(ast)} = ${text}`,
      answer: text,
      solution: text,
      steps,
    };
  }

  if (ast.kind === 'integral') {
    const anti = integrate(ast.body, ast.variable);
    if (!ast.from || !ast.to) {
      if (!anti) return symbolic(base, toLatex(ast), toText(ast), 'No elementary antiderivative was found for this integrand.');
      const text = `${toText(anti)} + C`;
      return {
        ...base,
        kind: 'evaluation',
        latex: `${toLatex(ast)} = ${toLatex(anti)} + C`,
        readableText: `${toText(ast)} = ${text}`,
        answer: text,
        solution: text,
        steps: ['Term by term: powers, elementary functions, parts or substitution', `= ${text}`],
      };
    }
    const freeBounds = containsVariable(ast.from) || containsVariable(ast.to);
    if (freeBounds) return symbolic(base, toLatex(ast), toText(ast), 'The limits are not numbers, so it can only be written.');
    const a = evaluate(ast.from);
    const b = evaluate(ast.to);
    const steps: string[] = [];
    let value: number;
    if (anti) {
      const F = (t: number) => evaluate(anti, { [ast.variable]: t });
      value = F(b) - F(a);
      steps.push(`Antiderivative: F(${ast.variable}) = ${toText(anti)}`, `F(${formatNumber(b)}) - F(${formatNumber(a)}) = ${formatNumber(F(b))} - ${formatNumber(F(a))}`);
    } else {
      value = evaluate(ast);
      steps.push('No closed form here, so it is computed numerically (Simpson rule, 2000 intervals)');
    }
    const answer = formatNumber(value);
    steps.push(`= ${answer}`);
    return {
      ...base,
      kind: 'evaluation',
      latex: `${toLatex(ast)} = ${answer}`,
      readableText: `${toText(ast)} = ${answer}`,
      answer,
      solution: answer,
      steps,
    };
  }
  return null;
}

/** What a name stands for on this page: "y" from y = swish(x), or "f" from f(x) = …, with the definitions applied. */
function namedExpression(name: string, defs: Definitions): { defined: Node; body: Node } | null {
  const variableDef = defs.variables.get(name);
  const functionDef = defs.functions.get(name);
  const defined = variableDef ?? functionDef?.body;
  if (!defined) return null;
  return { defined, body: inlineDefinitions(defined, defs) };
}

/** Which differentiation rules the expression needs, in words. */
function rulesUsed(node: Node, v: string): string[] {
  const rules = new Set<string>();
  const walk = (n: Node) => {
    const depends = (m: Node) => freeVariables(m).has(v);
    switch (n.kind) {
      case 'bin':
        if (n.op === '*' && depends(n.left) && depends(n.right)) rules.add('Product rule: (uw)′ = u′w + uw′');
        if (n.op === '/' && depends(n.right)) rules.add('Quotient rule: (u/w)′ = (u′w − uw′) / w²');
        if (n.op === '^') rules.add(depends(n.right) ? 'Power with a variable exponent: u^w = e^(w ln u)' : 'Power rule: (uⁿ)′ = n uⁿ⁻¹ u′');
        walk(n.left);
        walk(n.right);
        break;
      case 'neg':
        walk(n.arg);
        break;
      case 'call':
        if (n.fn === 'sigmoid') rules.add('Sigmoid: σ′(u) = σ(u)(1 − σ(u)) u′');
        else if (n.fn === 'swish') rules.add('Swish: (uσ(u))′ = σ(u) + uσ(u)(1 − σ(u))');
        else if (n.fn === 'tanh') rules.add('tanh′(u) = 1 − tanh²(u)');
        if (n.arg.kind !== 'var') rules.add('Chain rule: (f(u))′ = f′(u) u′');
        walk(n.arg);
        break;
      default:
        break;
    }
  };
  walk(node);
  return [...rules];
}

function solveRelation(base: SolveResult, op: RelOp, lhs: Node, rhs: Node): SolveResult {
  const latex = `${toLatex(lhs)} ${EQUATION_LATEX[op]} ${toLatex(rhs)}`;
  const readable = `${toText(lhs)} ${op} ${toText(rhs)}`;
  const vars = [...new Set([...freeVariables(lhs), ...freeVariables(rhs)])];

  if (vars.length === 0) {
    const a = evaluate(lhs);
    const b = evaluate(rhs);
    const eps = 1e-9 * (1 + Math.abs(a) + Math.abs(b));
    const holds: Record<RelOp, boolean> = {
      '=': Math.abs(a - b) < eps,
      '<': a < b,
      '>': a > b,
      '≤': a <= b + eps,
      '≥': a >= b - eps,
      '≠': Math.abs(a - b) >= eps,
      '≈': Math.abs(a - b) < 0.01 * (1 + Math.abs(a)),
    };
    const ok = holds[op];
    return {
      ...base,
      kind: 'evaluation',
      latex,
      readableText: readable,
      answer: ok ? '✓' : '✗',
      solution: ok ? (op === '=' ? 'The equality holds' : 'True') : `False: ${formatNumber(a)} ${op === '=' ? '≠' : 'does not satisfy ' + op} ${formatNumber(b)}`,
      steps: [...simplificationSteps(lhs), op, ...simplificationSteps(rhs)],
    };
  }

  if (op === '=' && vars.length === 1) {
    return { ...base, kind: 'equation', latex, readableText: readable, ...solvePolynomial(lhs, rhs, vars[0]) };
  }
  return symbolic(base, latex, readable, vars.length > 1 ? `It has several unknowns (${vars.join(', ')}), so it can only be written.` : 'Only equations with one unknown can be solved.');
}

/**
 * Solve f(x) = g(x) when h(x) = f(x) - g(x) is a polynomial of degree <= 2.
 * The coefficients are found by sampling h at x = -1, 0, 1 and verified at other points.
 */
function solvePolynomial(lhs: Node, rhs: Node, v: string): Pick<SolveResult, 'answer' | 'solution' | 'steps'> {
  const h = (x: number) => evaluate(lhs, { [v]: x }) - evaluate(rhs, { [v]: x });
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

  const steps = [`Move everything to one side: ${polyText(a, b, c, v)} = 0`];
  const eps = 1e-12;

  if (Math.abs(a) < eps) {
    if (Math.abs(b) < eps) {
      const always = Math.abs(c) < eps;
      steps.push(always ? `True for every ${v}` : 'No solution');
      return { answer: always ? `${v} ∈ ℝ` : '∅', solution: always ? 'Infinitely many solutions' : 'No solution', steps };
    }
    const x = -c / b;
    steps.push(`Linear equation: ${formatNumber(b)}·${v} = ${formatNumber(-c)}`);
    steps.push(`${v} = ${formatNumber(-c)} / ${formatNumber(b)} = ${formatNumber(x)}`);
    return { answer: `${v} = ${formatNumber(x)}`, solution: `${v} = ${formatNumber(x)}`, steps };
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
    steps.push(`Double root: ${v} = -b / 2a = ${formatNumber(x)}`);
    return { answer: `${v} = ${formatNumber(x)}`, solution: `${v} = ${formatNumber(x)}`, steps };
  }
  const sq = Math.sqrt(disc);
  const x1 = (-b - sq) / (2 * a);
  const x2 = (-b + sq) / (2 * a);
  const [lo, hi] = x1 < x2 ? [x1, x2] : [x2, x1];
  steps.push(`${v} = (-b ± √disc) / 2a → ${v}₁ = ${formatNumber(lo)}, ${v}₂ = ${formatNumber(hi)}`);
  const text = `${v} = ${formatNumber(lo)}, ${v} = ${formatNumber(hi)}`;
  return { answer: text, solution: text, steps };
}

function polyText(a: number, b: number, c: number, v: string): string {
  const parts: string[] = [];
  const add = (coef: number, suffix: string) => {
    if (Math.abs(coef) < 1e-12) return;
    const sign = coef < 0 ? '-' : '+';
    const abs = Math.abs(coef);
    const n = suffix && Math.abs(abs - 1) < 1e-12 ? '' : formatNumber(abs);
    parts.push(parts.length === 0 ? `${coef < 0 ? '-' : ''}${n}${suffix}` : ` ${sign} ${n}${suffix}`);
  };
  add(a, `${v}²`);
  add(b, v);
  add(c, '');
  return parts.length ? parts.join('') : '0';
}
