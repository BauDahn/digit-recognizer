/**
 * Output classes of each network. The index in the array is the output neuron.
 */

export const DIGIT_CLASSES = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'];

/**
 * EMNIST "Balanced": 47 classes. Letters whose upper and lower case look alike
 * (c, i, j, k, l, m, o, p, s, u, v, w, x, y, z) are merged into the uppercase class.
 */
export const EMNIST_BALANCED_CLASSES = [
  ...DIGIT_CLASSES,
  ...'ABCDEFGHIJKLMNOPQRSTUVWXYZ'.split(''),
  ...'abdefghnqrt'.split(''),
];

/**
 * Symbols the math canvas understands. Digits come from MNIST, 'x' from EMNIST and
 * synthetic strokes, and the operators from synthetic strokes (see scripts/lib/syntheticSymbols.ts).
 * Dots (decimal point / multiplication dot) are detected geometrically, not by the network.
 */
export const MATH_SYMBOL_CLASSES = [...DIGIT_CLASSES, '+', '-', '/', '=', '(', ')', 'x'];
