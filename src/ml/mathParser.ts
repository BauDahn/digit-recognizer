/**
 * Tokenizer and recursive-descent parser for the text the handwriting recognizer produces.
 *
 *   relation := expr (relop expr)?
 *   expr     := term (('+' | '-' | '±') term)*
 *   term     := unary (('*' | '/') unary | implicit multiplication)*
 *   unary    := ('-' | '+') unary | power
 *   power    := postfix ('^' unary)?
 *   postfix  := atom '!'*
 *   atom     := number | variable | constant | '(' expr ')' | '|' expr '|' | function argument
 *             | 'Σ'/'∏' ['_' group] ['^' group] term | '∫' ['_' group] ['^' group] expr 'd' variable
 *             | '√' atom | \frac(a,b) | \sqrt(a) | \root(n,a)
 *
 * Structure coming from the page layout is written as `^(…)`, `_(…)`, `\frac(a,b)` and `\sqrt(a)`.
 * A fraction like d/dx or ∂f/∂x becomes a derivative operator.
 */
import { bin, EMPTY, evaluate, MathSyntaxError, Node, num, RelOp, variable } from './mathAst';

type Token =
  | { type: 'num'; value: number }
  | { type: 'id'; name: string }
  | { type: 'const'; name: 'π' | 'e' | '∞' }
  | { type: 'fn'; name: string }
  | { type: 'big'; name: 'Σ' | '∏' | '∫' }
  | { type: 'cmd'; name: 'frac' | 'sqrt' | 'root' }
  | { type: 'op'; value: string };

// longer names first: 'sinh' before 'sin'
const FUNCTION_NAMES = ['softplus', 'sigmoid', 'arcsin', 'arccos', 'arctan', 'swish', 'sqrt', 'sinh', 'cosh', 'tanh', 'sin', 'cos', 'tan', 'cot', 'sec', 'csc', 'ln', 'log', 'exp'];
const SUPERSCRIPT_DIGITS = '⁰¹²³⁴⁵⁶⁷⁸⁹';
const OPS = '+-*/^_()=<>!,|%';
const OP_ALIASES: Record<string, string> = {
  '×': '*',
  '·': '*',
  '∘': '*',
  '÷': '/',
  '−': '-',
  '[': '(',
  ']': ')',
  '{': '(',
  '}': ')',
  '⟨': '(',
  '⟩': ')',
  '≤': '≤',
  '≥': '≥',
  '≠': '≠',
  '≈': '≈',
  '±': '±',
};
const GREEK = /[α-ωΑ-Ω]/;

/**
 * `userFunctions` are names the page defines itself (f(x) = …, swish(x) = …): they are functions
 * wherever they are followed by '('. σ followed by '(' is the sigmoid.
 */
export function tokenize(input: string, userFunctions: ReadonlySet<string> = new Set()): Token[] {
  const tokens: Token[] = [];
  const src = input.replace(/\s+/g, '');
  const userNames = [...userFunctions].sort((a, b) => b.length - a.length);
  let i = 0;
  while (i < src.length) {
    const ch = src[i];
    const user = userNames.find((name) => src.startsWith(name, i) && src[i + name.length] === '(');
    if (user) {
      tokens.push({ type: 'fn', name: user });
      i += user.length;
    } else if (/[0-9.]/.test(ch)) {
      let j = i;
      while (j < src.length && /[0-9.]/.test(src[j])) j++;
      const text = src.slice(i, j);
      if ((text.match(/\./g) || []).length > 1 || text === '.') throw new MathSyntaxError(`Invalid number: ${text}`);
      tokens.push({ type: 'num', value: parseFloat(text) });
      i = j;
    } else if (SUPERSCRIPT_DIGITS.includes(ch)) {
      // x² is x^(2)
      let j = i;
      let digits = '';
      while (j < src.length && SUPERSCRIPT_DIGITS.includes(src[j])) digits += SUPERSCRIPT_DIGITS.indexOf(src[j++]);
      tokens.push({ type: 'op', value: '^' }, { type: 'op', value: '(' }, { type: 'num', value: Number(digits) }, { type: 'op', value: ')' });
      i = j;
    } else if (ch === '\\') {
      const m = /^\\(frac|sqrt|root)/.exec(src.slice(i));
      if (!m) throw new MathSyntaxError('Unknown command');
      tokens.push({ type: 'cmd', name: m[1] as 'frac' | 'sqrt' | 'root' });
      i += m[0].length;
    } else if (/[A-Za-z]/.test(ch)) {
      const fn = FUNCTION_NAMES.find((name) => src.startsWith(name, i));
      if (fn) {
        tokens.push({ type: 'fn', name: fn });
        i += fn.length;
      } else if (ch === 'e') {
        tokens.push({ type: 'const', name: 'e' });
        i++;
      } else {
        tokens.push({ type: 'id', name: ch });
        i++;
      }
    } else if (ch === 'π') {
      tokens.push({ type: 'const', name: 'π' });
      i++;
    } else if (ch === '∞') {
      tokens.push({ type: 'const', name: '∞' });
      i++;
    } else if (ch === 'Σ' || ch === '∑' || ch === '∏' || ch === '∫' || ch === '∮') {
      tokens.push({ type: 'big', name: ch === '∮' ? '∫' : ch === '∑' ? 'Σ' : ch });
      i++;
    } else if (ch === 'σ' && src[i + 1] === '(') {
      tokens.push({ type: 'fn', name: 'sigmoid' });
      i++;
    } else if (GREEK.test(ch) || ch === '∂' || ch === 'ℓ') {
      tokens.push({ type: 'id', name: ch });
      i++;
    } else if (ch === '√') {
      tokens.push({ type: 'cmd', name: 'sqrt' });
      i++;
    } else if (ch === '⌊' || ch === '⌋' || ch === '⌈' || ch === '⌉') {
      tokens.push({ type: 'op', value: ch });
      i++;
    } else if (OPS.includes(ch) || OP_ALIASES[ch]) {
      tokens.push({ type: 'op', value: OP_ALIASES[ch] ?? ch });
      i++;
    } else {
      throw new MathSyntaxError(`Cannot compute with the symbol "${ch}"`);
    }
  }
  return tokens;
}

const REL_OPS = new Set(['=', '<', '>', '≤', '≥', '≠', '≈']);

class Parser {
  private pos = 0;
  /** Inside |…| a second '|' closes instead of opening. */
  private absDepth = 0;
  /** Inside an integral, "d x" ends the integrand. */
  private integralDepth = 0;

  constructor(private tokens: Token[]) {}

  atEnd(): boolean {
    return this.pos >= this.tokens.length;
  }

  private peek(offset = 0): Token | undefined {
    return this.tokens[this.pos + offset];
  }

  private isOp(value: string, offset = 0): boolean {
    const t = this.peek(offset);
    return t?.type === 'op' && t.value === value;
  }

  private expectOp(value: string) {
    if (!this.isOp(value)) throw new MathSyntaxError(`Expected "${value}"`);
    this.pos++;
  }

  /** relation := expr (relop expr)? ; the right side may be empty ("12+7=") */
  parseRelation(): Node {
    const left = this.parseExpr();
    const t = this.peek();
    if (t?.type === 'op' && REL_OPS.has(t.value)) {
      this.pos++;
      if (this.atEnd()) return { kind: 'rel', op: t.value as RelOp, left, right: EMPTY };
      return { kind: 'rel', op: t.value as RelOp, left, right: this.parseExpr() };
    }
    return left;
  }

  parseExpr(): Node {
    let node = this.parseTerm();
    while (this.isOp('+') || this.isOp('-') || this.isOp('±')) {
      const op = (this.tokens[this.pos++] as { value: '+' | '-' | '±' }).value;
      node = bin(op, node, this.parseTerm());
    }
    return node;
  }

  private parseTerm(): Node {
    let node = this.parseUnary();
    for (;;) {
      if (this.isOp('*') || this.isOp('/')) {
        const op = (this.tokens[this.pos++] as { value: '*' | '/' }).value;
        node = bin(op, node, this.parseUnary());
      } else if (this.startsAtom()) {
        // Implicit multiplication: 2x, 3(x+1), (x+1)(x-1)
        node = bin('*', node, this.parsePower());
      } else {
        return node;
      }
    }
  }

  private isDifferentialAhead(): boolean {
    const t = this.peek();
    const next = this.peek(1);
    return t?.type === 'id' && (t.name === 'd' || t.name === 'δ') && next?.type === 'id';
  }

  private startsAtom(): boolean {
    const t = this.peek();
    if (!t) return false;
    if (this.integralDepth > 0 && this.isDifferentialAhead()) return false;
    switch (t.type) {
      case 'num':
      case 'id':
      case 'const':
      case 'fn':
      case 'big':
      case 'cmd':
        return true;
      case 'op':
        if (t.value === '|') return this.absDepth === 0;
        return t.value === '(' || t.value === '⌊' || t.value === '⌈';
    }
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
    let base = this.parsePostfix();
    // a subscript may come before the exponent: x_1^2
    if (this.isOp('_')) {
      this.pos++;
      const sub = this.parseScriptGroup();
      base = this.attachSubscript(base, sub);
    }
    if (this.isOp('^')) {
      this.pos++;
      return bin('^', base, this.parseScriptGroup(true));
    }
    return base;
  }

  /** Content of a ^ or _ : a parenthesized group, or a single atom. */
  private parseScriptGroup(allowUnary = false): Node {
    if (this.isOp('(')) {
      this.pos++;
      const inner = this.parseExpr();
      this.expectOp(')');
      return inner;
    }
    if (allowUnary) return this.parseUnary();
    return this.parsePostfix();
  }

  private attachSubscript(base: Node, sub: Node): Node {
    if (base.kind !== 'var') throw new MathSyntaxError('Only variables can have a subscript');
    const text = sub.kind === 'num' ? String(sub.value) : sub.kind === 'var' ? sub.name : null;
    if (text === null) {
      // subscript like (n+1): flatten to a readable name
      return variable(`${base.name}_${flattenSubscript(sub)}`);
    }
    return variable(`${base.name}_${text}`);
  }

  private parsePostfix(): Node {
    let node = this.parseAtom();
    while (this.isOp('!')) {
      this.pos++;
      node = { kind: 'fact', arg: node };
    }
    if (this.isOp('%')) {
      this.pos++;
      node = bin('/', node, num(100));
    }
    return node;
  }

  private parseGroup(): Node {
    this.expectOp('(');
    const inner = this.parseExpr();
    this.expectOp(')');
    return inner;
  }

  private parseAtom(): Node {
    const t = this.peek();
    if (!t) throw new MathSyntaxError('The expression is incomplete');
    this.pos++;

    switch (t.type) {
      case 'num':
        return num(t.value);
      case 'const':
        return { kind: 'const', name: t.name, value: t.name === 'π' ? Math.PI : t.name === 'e' ? Math.E : Infinity };
      case 'id':
        return variable(t.name);
      case 'fn':
        return this.parseFunction(t.name);
      case 'big':
        return this.parseBigOperator(t.name);
      case 'cmd':
        return this.parseCommand(t.name);
      case 'op': {
        if (t.value === '(') {
          this.pos--;
          return this.parseGroup();
        }
        if (t.value === '|') {
          this.absDepth++;
          const inner = this.parseExpr();
          this.absDepth--;
          this.expectOp('|');
          return { kind: 'call', fn: 'abs', arg: inner };
        }
        if (t.value === '⌊' || t.value === '⌈') {
          const close = t.value === '⌊' ? '⌋' : '⌉';
          const inner = this.parseExpr();
          this.expectOp(close);
          return { kind: 'call', fn: t.value === '⌊' ? 'floor' : 'ceil', arg: inner };
        }
        throw new MathSyntaxError(`Unexpected symbol: ${t.value}`);
      }
    }
  }

  /** sin x, sin(x), sin 2x, sin^2 x */
  private parseFunction(fn: string): Node {
    let exponent: Node | null = null;
    if (this.isOp('^')) {
      this.pos++;
      exponent = this.parseScriptGroup();
    }
    let arg: Node;
    if (this.isOp('(')) {
      arg = this.parseGroup();
    } else {
      // the run of factors written right after the name: sin 2x = sin(2x)
      arg = this.parsePower();
      while (this.startsAtom() && this.peek()?.type !== 'fn' && this.peek()?.type !== 'big') {
        arg = bin('*', arg, this.parsePower());
      }
    }
    const call: Node = { kind: 'call', fn, arg };
    return exponent ? bin('^', call, exponent) : call;
  }

  private parseCommand(name: 'frac' | 'sqrt' | 'root'): Node {
    // √ written without a group: √x, √2
    if (name === 'sqrt' && !this.isOp('(')) return { kind: 'call', fn: 'sqrt', arg: this.parsePower() };
    this.expectOp('(');
    const first = this.parseExpr();
    if (name === 'sqrt') {
      this.expectOp(')');
      return { kind: 'call', fn: 'sqrt', arg: first };
    }
    this.expectOp(',');
    const second = this.parseExpr();
    this.expectOp(')');
    if (name === 'root') {
      return bin('^', second, bin('/', num(1), first));
    }
    return this.fractionOrDerivative(first, second);
  }

  private fractionOrDerivative(numerator: Node, denominator: Node): Node {
    const deriv = matchDerivative(numerator, denominator);
    if (!deriv) return bin('/', numerator, denominator);
    if (!deriv.fn && !deriv.body && this.startsAtom()) {
      // d/dx (x^2 + 1): the operator applies to what follows
      return { kind: 'deriv', ...deriv, body: this.parsePower() };
    }
    return { kind: 'deriv', ...deriv };
  }

  private parseBigOperator(op: 'Σ' | '∏' | '∫'): Node {
    let lower: Node | undefined;
    let upper: Node | undefined;
    for (let i = 0; i < 2; i++) {
      if (this.isOp('_') && !lower) {
        this.pos++;
        lower = this.parseScriptLimit();
      } else if (this.isOp('^') && !upper) {
        this.pos++;
        upper = this.parseScriptLimit();
      }
    }

    if (op === '∫') {
      this.integralDepth++;
      const body = this.parseExprUntilDifferential();
      this.integralDepth--;
      const d = this.peek();
      const v = this.peek(1);
      if (d?.type !== 'id' || (d.name !== 'd' && d.name !== 'δ') || v?.type !== 'id') {
        throw new MathSyntaxError('An integral must end with "dx" (or dt, dy…)');
      }
      this.pos += 2;
      return { kind: 'integral', variable: v.name, from: lower, to: upper, body };
    }

    if (!lower || !upper) throw new MathSyntaxError(`${op} needs a lower and an upper limit, like Σ below i=1 and n above`);
    if (lower.kind !== 'rel' || lower.op !== '=' || lower.left.kind !== 'var') {
      throw new MathSyntaxError(`The lower limit of ${op} must look like i = 1`);
    }
    const body = this.parseTerm();
    return { kind: 'bigop', op, index: lower.left.name, from: lower.right, to: upper, body };
  }

  /** Limit of Σ/∏/∫: i=1, n, -∞ ... as a group or a single atom. */
  private parseScriptLimit(): Node {
    if (this.isOp('(')) {
      this.pos++;
      const inner = this.parseRelation();
      this.expectOp(')');
      return inner;
    }
    return this.parseUnary();
  }

  private parseExprUntilDifferential(): Node {
    return this.parseExpr();
  }
}

function flattenSubscript(node: Node): string {
  switch (node.kind) {
    case 'num':
      return String(node.value);
    case 'var':
      return node.name;
    case 'bin':
      return `${flattenSubscript(node.left)}${node.op}${flattenSubscript(node.right)}`;
    default:
      return '?';
  }
}

function factors(node: Node): Node[] {
  if (node.kind === 'bin' && node.op === '*') return [...factors(node.left), ...factors(node.right)];
  return [node];
}

/**
 * d/dx, ∂/∂x, d²/dx², ∂f/∂x, ∂f(x)/∂x, ∂²f/∂x∂y ... as a (numerator, denominator) pair of expressions.
 * `variables` is only set for mixed partial derivatives.
 */
function matchDerivative(
  numerator: Node,
  denominator: Node
): { variable: string; variables?: string[]; partial: boolean; order: number; fn?: string; body?: Node } | null {
  const symbolOf = (n: Node): { partial: boolean; order: number } | null => {
    const base = n.kind === 'bin' && n.op === '^' ? n.left : n;
    if (base.kind !== 'var' || (base.name !== 'd' && base.name !== '∂')) return null;
    const order = n.kind === 'bin' && n.op === '^' && n.right.kind === 'num' ? n.right.value : 1;
    return { partial: base.name === '∂', order };
  };

  const top = factors(numerator);
  const bottom = factors(denominator);
  const head = symbolOf(top[0]);
  if (!head || top.length > 2) return null;

  // denominator: dx, dx², ∂x∂y … (a differential symbol followed by its variable, repeated)
  const variables: string[] = [];
  if (bottom.length % 2 !== 0) return null;
  for (let i = 0; i < bottom.length; i += 2) {
    const symbol = symbolOf(bottom[i]);
    if (!symbol || symbol.partial !== head.partial) return null;
    const v = bottom[i + 1];
    const base = v.kind === 'bin' && v.op === '^' ? v.left : v;
    if (base.kind !== 'var') return null;
    const times = v.kind === 'bin' && v.op === '^' && v.right.kind === 'num' ? v.right.value : 1;
    for (let k = 0; k < times; k++) variables.push(base.name);
  }
  if (variables.length === 0 || variables.length !== head.order) return null;

  let fn: string | undefined;
  let body: Node | undefined;
  if (top.length === 2) {
    if (top[1].kind === 'var') fn = top[1].name;
    else if (top[1].kind === 'call') body = top[1];
    else return null;
  }
  const mixed = variables.some((v) => v !== variables[0]);
  return { variable: variables[0], variables: mixed ? variables : undefined, partial: head.partial, order: head.order, fn, body };
}

export function parseRelation(input: string, userFunctions?: ReadonlySet<string>): Node {
  const parser = new Parser(tokenize(input, userFunctions));
  const node = parser.parseRelation();
  if (!parser.atEnd()) throw new MathSyntaxError('Unexpected symbols at the end of the expression');
  return node;
}

export function parseExpression(input: string, userFunctions?: ReadonlySet<string>): Node {
  const parser = new Parser(tokenize(input, userFunctions));
  const node = parser.parseExpr();
  if (!parser.atEnd()) throw new MathSyntaxError('Unexpected symbols at the end of the expression');
  return node;
}

/** Compile an expression in x into a plain function, for plotting. */
export function compileFunction(input: string): (x: number) => number {
  const ast = parseExpression(input);
  return (x: number) => evaluate(ast, { x });
}
