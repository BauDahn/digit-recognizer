/**
 * Catalogue of every character the "math" network can recognize.
 * The order is the order of the output neurons, so it must match the trained model
 * (public/models/math.json stores its own class list too, this is what the UI uses
 * to explain symbols and to let people pick the right one when the network is wrong).
 */

export type SymbolGroup =
  | 'digits'
  | 'letters'
  | 'greek'
  | 'calculus'
  | 'operators'
  | 'relations'
  | 'sets'
  | 'arrows'
  | 'brackets'
  | 'misc';

export interface SymbolInfo {
  /** What the network outputs and what the parser reads. */
  char: string;
  /** LaTeX that renders this symbol on its own. */
  latex: string;
  name: string;
  group: SymbolGroup;
}

export const SYMBOL_GROUP_LABELS: Record<SymbolGroup, string> = {
  digits: 'Digits',
  letters: 'Letters',
  greek: 'Greek',
  calculus: 'Calculus',
  operators: 'Operators',
  relations: 'Relations',
  sets: 'Sets & logic',
  arrows: 'Arrows',
  brackets: 'Brackets',
  misc: 'Other',
};

const s = (group: SymbolGroup, char: string, latex: string, name: string): SymbolInfo => ({ char, latex, name, group });

const DIGITS: SymbolInfo[] = '0123456789'.split('').map((d) => s('digits', d, d, `digit ${d}`));

/**
 * Letters: where upper and lower case look alike (c, j, k, m, o, p, s, u, v, w, x, y, z)
 * there is a single class written in lower case. I, l and | are left out on purpose:
 * on their own they are all just a vertical bar, indistinguishable from 1.
 */
const LETTERS: SymbolInfo[] = [
  ...'ABDEFGHLNQRT'.split('').map((c) => s('letters', c, c, `capital ${c}`)),
  ...'abdefghinqrt'.split('').map((c) => s('letters', c, c, `letter ${c}`)),
  ...'cjkmopsuvwxyz'.split('').map((c) => s('letters', c, c, `letter ${c}`)),
];

const GREEK: SymbolInfo[] = [
  s('greek', 'α', '\\alpha', 'alpha'),
  s('greek', 'β', '\\beta', 'beta'),
  s('greek', 'γ', '\\gamma', 'gamma'),
  s('greek', 'δ', '\\delta', 'delta'),
  s('greek', 'ε', '\\varepsilon', 'epsilon'),
  s('greek', 'ζ', '\\zeta', 'zeta'),
  s('greek', 'η', '\\eta', 'eta'),
  s('greek', 'θ', '\\theta', 'theta'),
  s('greek', 'κ', '\\kappa', 'kappa'),
  s('greek', 'λ', '\\lambda', 'lambda'),
  s('greek', 'μ', '\\mu', 'mu'),
  s('greek', 'ν', '\\nu', 'nu'),
  s('greek', 'ξ', '\\xi', 'xi'),
  s('greek', 'π', '\\pi', 'pi'),
  s('greek', 'ρ', '\\rho', 'rho'),
  s('greek', 'σ', '\\sigma', 'sigma'),
  s('greek', 'τ', '\\tau', 'tau'),
  s('greek', 'φ', '\\varphi', 'phi'),
  s('greek', 'χ', '\\chi', 'chi'),
  s('greek', 'ψ', '\\psi', 'psi'),
  s('greek', 'ω', '\\omega', 'omega'),
  s('greek', 'Γ', '\\Gamma', 'capital Gamma'),
  s('greek', 'Θ', '\\Theta', 'capital Theta'),
  s('greek', 'Λ', '\\Lambda', 'capital Lambda'),
  s('greek', 'Ξ', '\\Xi', 'capital Xi'),
  s('greek', 'Φ', '\\Phi', 'capital Phi'),
  s('greek', 'Ψ', '\\Psi', 'capital Psi'),
  s('greek', 'Ω', '\\Omega', 'capital Omega'),
];

const CALCULUS: SymbolInfo[] = [
  s('calculus', '∂', '\\partial', 'partial derivative'),
  s('calculus', '∇', '\\nabla', 'nabla (gradient)'),
  s('calculus', '∫', '\\int', 'integral'),
  s('calculus', '∮', '\\oint', 'contour integral'),
  s('calculus', 'Σ', '\\sum', 'summation (sigma)'),
  s('calculus', '∏', '\\prod', 'product (pi)'),
  s('calculus', '∞', '\\infty', 'infinity'),
  s('calculus', '√', '\\sqrt{}', 'square root'),
  s('calculus', 'Δ', '\\Delta', 'delta (change)'),
];

const OPERATORS: SymbolInfo[] = [
  s('operators', '+', '+', 'plus'),
  s('operators', '-', '-', 'minus'),
  s('operators', '×', '\\times', 'times'),
  s('operators', '÷', '\\div', 'divided by'),
  s('operators', '/', '/', 'slash'),
  s('operators', '±', '\\pm', 'plus-minus'),
  s('operators', '∘', '\\circ', 'composition'),
];

const RELATIONS: SymbolInfo[] = [
  s('relations', '=', '=', 'equals'),
  s('relations', '≠', '\\neq', 'not equal'),
  s('relations', '≈', '\\approx', 'approximately'),
  s('relations', '≡', '\\equiv', 'equivalent'),
  s('relations', '∼', '\\sim', 'similar'),
  s('relations', '≅', '\\cong', 'congruent'),
  s('relations', '<', '<', 'less than'),
  s('relations', '>', '>', 'greater than'),
  s('relations', '≤', '\\leq', 'less or equal'),
  s('relations', '≥', '\\geq', 'greater or equal'),
  s('relations', '∝', '\\propto', 'proportional'),
];

const SETS: SymbolInfo[] = [
  s('sets', '∈', '\\in', 'element of'),
  s('sets', '∉', '\\notin', 'not element of'),
  s('sets', '⊂', '\\subset', 'subset'),
  s('sets', '⊆', '\\subseteq', 'subset or equal'),
  s('sets', '⊃', '\\supset', 'superset'),
  s('sets', '⊇', '\\supseteq', 'superset or equal'),
  s('sets', '∪', '\\cup', 'union'),
  s('sets', '∩', '\\cap', 'intersection'),
  s('sets', '∅', '\\emptyset', 'empty set'),
  s('sets', '∀', '\\forall', 'for all'),
  s('sets', '∃', '\\exists', 'there exists'),
  s('sets', '∄', '\\nexists', 'does not exist'),
  s('sets', '∧', '\\wedge', 'and'),
  s('sets', '∨', '\\vee', 'or'),
  s('sets', '¬', '\\neg', 'not'),
  s('sets', '⊥', '\\perp', 'perpendicular'),
  s('sets', '∥', '\\parallel', 'parallel'),
  s('sets', '∴', '\\therefore', 'therefore'),
  s('sets', '∵', '\\because', 'because'),
  s('sets', '⊕', '\\oplus', 'direct sum'),
  s('sets', '⊗', '\\otimes', 'tensor product'),
  s('sets', 'ℝ', '\\mathbb{R}', 'real numbers'),
  s('sets', 'ℕ', '\\mathbb{N}', 'natural numbers'),
  s('sets', 'ℤ', '\\mathbb{Z}', 'integers'),
  s('sets', 'ℂ', '\\mathbb{C}', 'complex numbers'),
  s('sets', 'ℚ', '\\mathbb{Q}', 'rationals'),
  s('sets', 'ℓ', '\\ell', 'script l'),
  s('sets', 'ℏ', '\\hbar', 'h-bar'),
  s('sets', 'ℵ', '\\aleph', 'aleph'),
];

const ARROWS: SymbolInfo[] = [
  s('arrows', '→', '\\to', 'right arrow'),
  s('arrows', '←', '\\leftarrow', 'left arrow'),
  s('arrows', '⇒', '\\Rightarrow', 'implies'),
  s('arrows', '⇐', '\\Leftarrow', 'implied by'),
  s('arrows', '⇔', '\\Leftrightarrow', 'if and only if'),
  s('arrows', '↔', '\\leftrightarrow', 'two-way arrow'),
  s('arrows', '↦', '\\mapsto', 'maps to'),
  s('arrows', '↑', '\\uparrow', 'up arrow'),
  s('arrows', '↓', '\\downarrow', 'down arrow'),
];

const BRACKETS: SymbolInfo[] = [
  s('brackets', '(', '(', 'left parenthesis'),
  s('brackets', ')', ')', 'right parenthesis'),
  s('brackets', '[', '[', 'left bracket'),
  s('brackets', ']', ']', 'right bracket'),
  s('brackets', '{', '\\{', 'left brace'),
  s('brackets', '}', '\\}', 'right brace'),
  s('brackets', '⟨', '\\langle', 'left angle bracket'),
  s('brackets', '⟩', '\\rangle', 'right angle bracket'),
  s('brackets', '⌊', '\\lfloor', 'left floor'),
  s('brackets', '⌋', '\\rfloor', 'right floor'),
  s('brackets', '⌈', '\\lceil', 'left ceiling'),
  s('brackets', '⌉', '\\rceil', 'right ceiling'),
  s('brackets', '|', '|', 'vertical bar'),
  s('brackets', '‖', '\\|', 'double bar (norm)'),
];

const MISC: SymbolInfo[] = [
  s('misc', '!', '!', 'factorial'),
  s('misc', '%', '\\%', 'percent'),
  s('misc', '&', '\\&', 'ampersand'),
  s('misc', '#', '\\#', 'hash'),
  s('misc', '$', '\\$', 'dollar'),
  s('misc', '°', '^\\circ', 'degree'),
  s('misc', '∠', '\\angle', 'angle'),
  s('misc', '△', '\\triangle', 'triangle'),
  s('misc', '\\', '\\backslash', 'backslash'),
];

export const SYMBOLS: SymbolInfo[] = [
  ...DIGITS,
  ...LETTERS,
  ...GREEK,
  ...CALCULUS,
  ...OPERATORS,
  ...RELATIONS,
  ...SETS,
  ...ARROWS,
  ...BRACKETS,
  ...MISC,
];

/** Output neurons of the math network, in order. */
export const MATH_CLASSES: string[] = SYMBOLS.map((sym) => sym.char);

const BY_CHAR = new Map(SYMBOLS.map((sym) => [sym.char, sym]));

export function symbolInfo(char: string): SymbolInfo | undefined {
  return BY_CHAR.get(char);
}

/** LaTeX for a recognized character (falls back to the character itself). */
export function symbolLatex(char: string): string {
  if (char === '*') return '\\cdot';
  return BY_CHAR.get(char)?.latex ?? char;
}

export function symbolsByGroup(): Array<{ group: SymbolGroup; label: string; symbols: SymbolInfo[] }> {
  const groups = new Map<SymbolGroup, SymbolInfo[]>();
  for (const sym of SYMBOLS) {
    if (!groups.has(sym.group)) groups.set(sym.group, []);
    groups.get(sym.group)!.push(sym);
  }
  return [...groups.entries()].map(([group, symbols]) => ({ group, label: SYMBOL_GROUP_LABELS[group], symbols }));
}
