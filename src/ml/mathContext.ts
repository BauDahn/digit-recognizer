/**
 * What the lines of one page define for each other:
 *
 *   swish(x) = xσ(βx)     a function (any name, one parameter)
 *   y = swish(x)          a name for an expression, used by ∂y/∂x
 *
 * Definitions can be written above or below the line that uses them.
 */
import { freeVariables, isEmpty, MathSyntaxError, Node, substitute } from './mathAst';
import { parseRelation } from './mathParser';

export interface FunctionDefinition {
  name: string;
  param: string;
  body: Node;
}

export interface Definitions {
  functions: Map<string, FunctionDefinition>;
  /** y = expression with unknowns, by the name y */
  variables: Map<string, Node>;
}

export const NO_DEFINITIONS: Definitions = { functions: new Map(), variables: new Map() };

/** Names that keep their meaning: nobody can redefine sin or ln. */
const RESERVED = new Set(['sin', 'cos', 'tan', 'cot', 'sec', 'csc', 'arcsin', 'arccos', 'arctan', 'ln', 'log', 'exp', 'sqrt', 'abs', 'floor', 'ceil', 'sinh', 'cosh', 'tanh', 'e', 'π', 'd', '∂']);

const clean = (text: string) => text.replace(/\s+/g, '').replace(/\?+$/, '');
const LETTER = '[A-Za-z\\u03B1-\\u03C9]';
const DEFINES_FUNCTION = new RegExp(`^(${LETTER}+)\\((${LETTER}(?:_[A-Za-z0-9])?)\\)=.`);

/** The function a line defines, if it starts like "name(param)=…". */
export function definedFunctionName(line: string): string | null {
  const m = DEFINES_FUNCTION.exec(clean(line));
  return m && !RESERVED.has(m[1]) ? m[1] : null;
}

/** Collect the definitions from the text of every line of a page. */
export function collectDefinitions(lines: string[]): Definitions {
  const names = new Set(lines.map(definedFunctionName).filter((n): n is string => n !== null));
  const functions = new Map<string, FunctionDefinition>();
  const variables = new Map<string, Node>();
  for (const line of lines) {
    let ast: Node;
    try {
      ast = parseRelation(clean(line), names);
    } catch {
      continue;
    }
    if (ast.kind !== 'rel' || ast.op !== '=' || isEmpty(ast.right)) continue;
    const { left, right } = ast;
    if (left.kind === 'call' && names.has(left.fn) && left.arg.kind === 'var') {
      functions.set(left.fn, { name: left.fn, param: left.arg.name, body: right });
    } else if (left.kind === 'var' && left.name) {
      const unknowns = freeVariables(right);
      if (unknowns.size > 0 && !unknowns.has(left.name)) variables.set(left.name, right);
    }
  }
  return { functions, variables };
}

/** Replace every call of a defined function by its body, e.g. swish(2t) → 2t·σ(β·2t). */
export function inlineDefinitions(node: Node, defs: Definitions, depth = 0): Node {
  if (defs.functions.size === 0) return node;
  if (depth > 16) throw new MathSyntaxError('A function is defined in terms of itself');
  const go = (n: Node): Node => inlineDefinitions(n, defs, depth);
  switch (node.kind) {
    case 'call': {
      const def = defs.functions.get(node.fn);
      const arg = go(node.arg);
      if (!def) return { ...node, arg };
      return inlineDefinitions(substitute(def.body, def.param, arg), defs, depth + 1);
    }
    case 'neg':
    case 'fact':
      return { ...node, arg: go(node.arg) };
    case 'bin':
    case 'rel':
      return { ...node, left: go(node.left), right: go(node.right) } as Node;
    case 'bigop':
      return { ...node, from: go(node.from), to: go(node.to), body: go(node.body) };
    case 'integral':
      return { ...node, from: node.from && go(node.from), to: node.to && go(node.to), body: go(node.body) };
    case 'deriv':
      return { ...node, body: node.body && go(node.body) };
    default:
      return node;
  }
}
