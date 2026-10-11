import { NeuralNetwork } from './neuralNetwork';
import type { NetworkWeights } from './trainingClient';

/** Numbers that describe a network: what the lab shows next to the diagram. Pure functions. */

export interface LayerSummary {
  name: string;
  inputs: number;
  outputs: number;
  weightCount: number;
  biasCount: number;
  params: number;
  mean: number;
  std: number;
  min: number;
  max: number;
  absMean: number;
  /** Euclidean norm of the weights */
  l2: number;
  /** Fraction of weights with |w| < 0.01 */
  nearZero: number;
  biasMean: number;
  /** Counts per bin of a histogram that goes from -range to +range */
  histogram: number[];
  histogramRange: number;
}

export interface NetworkSummary {
  layers: LayerSummary[];
  totalParams: number;
  /** Size of the weights as 32-bit floats */
  bytes: number;
  /** Multiply-adds needed to classify one image (dense, ignoring the zero pixels) */
  macs: number;
}

const BINS = 31;

function summarizeLayer(name: string, w: Float32Array, b: Float32Array, inputs: number, outputs: number): LayerSummary {
  let sum = 0;
  let sumAbs = 0;
  let sumSq = 0;
  let min = Infinity;
  let max = -Infinity;
  let near = 0;
  for (let i = 0; i < w.length; i++) {
    const v = w[i];
    sum += v;
    sumAbs += Math.abs(v);
    sumSq += v * v;
    if (v < min) min = v;
    if (v > max) max = v;
    if (Math.abs(v) < 0.01) near++;
  }
  const n = Math.max(1, w.length);
  const mean = sum / n;
  const std = Math.sqrt(Math.max(0, sumSq / n - mean * mean));
  // histogram range: 4 standard deviations around zero, so the bell of a fresh network is visible
  const range = Math.max(0.05, 4 * Math.sqrt(sumSq / n));
  const histogram = new Array(BINS).fill(0);
  for (let i = 0; i < w.length; i++) {
    const bin = Math.floor(((w[i] + range) / (2 * range)) * BINS);
    histogram[Math.max(0, Math.min(BINS - 1, bin))]++;
  }
  let biasSum = 0;
  for (let i = 0; i < b.length; i++) biasSum += b[i];
  return {
    name,
    inputs,
    outputs,
    weightCount: w.length,
    biasCount: b.length,
    params: w.length + b.length,
    mean,
    std,
    min,
    max,
    absMean: sumAbs / n,
    l2: Math.sqrt(sumSq),
    nearZero: near / n,
    biasMean: biasSum / Math.max(1, b.length),
    histogram,
    histogramRange: range,
  };
}

export function summarizeNetwork(net: NeuralNetwork): NetworkSummary {
  const { inputSize, hiddenSize1, hiddenSize2, outputSize } = net.config;
  const layers = [
    summarizeLayer('Hidden 1', net.w1, net.b1, inputSize, hiddenSize1),
    summarizeLayer('Hidden 2', net.w2, net.b2, hiddenSize1, hiddenSize2),
    summarizeLayer('Output', net.w3, net.b3, hiddenSize2, outputSize),
  ];
  const totalParams = layers.reduce((a, l) => a + l.params, 0);
  return {
    layers,
    totalParams,
    bytes: totalParams * 4,
    macs: layers.reduce((a, l) => a + l.weightCount, 0),
  };
}

export interface ClassMetric {
  index: number;
  label: string;
  precision: number;
  recall: number;
  f1: number;
  support: number;
  correct: number;
}

export interface ClassReport {
  perClass: ClassMetric[];
  accuracy: number;
  macroF1: number;
  weightedF1: number;
  total: number;
}

/** Precision, recall and F1 per class from a confusion matrix indexed [actual][predicted]. */
export function classReport(confusion: number[][], labels: string[]): ClassReport {
  const n = confusion.length;
  const predictedTotals = new Array(n).fill(0);
  for (let a = 0; a < n; a++) for (let p = 0; p < n; p++) predictedTotals[p] += confusion[a][p];

  let total = 0;
  let correct = 0;
  const perClass: ClassMetric[] = [];
  for (let i = 0; i < n; i++) {
    const support = confusion[i].reduce((x, y) => x + y, 0);
    const tp = confusion[i][i];
    total += support;
    correct += tp;
    const precision = predictedTotals[i] > 0 ? tp / predictedTotals[i] : 0;
    const recall = support > 0 ? tp / support : 0;
    const f1 = precision + recall > 0 ? (2 * precision * recall) / (precision + recall) : 0;
    perClass.push({ index: i, label: labels[i] ?? String(i), precision, recall, f1, support, correct: tp });
  }
  const withData = perClass.filter((c) => c.support > 0);
  const macroF1 = withData.reduce((a, c) => a + c.f1, 0) / Math.max(1, withData.length);
  const weightedF1 = total > 0 ? withData.reduce((a, c) => a + c.f1 * c.support, 0) / total : 0;
  return { perClass, accuracy: total > 0 ? correct / total : 0, macroF1, weightedF1, total };
}

export interface Confusion {
  actual: string;
  predicted: string;
  count: number;
}

/** The mistakes made most often, most frequent first. */
export function topConfusions(confusion: number[][], labels: string[], limit = 8): Confusion[] {
  const out: Confusion[] = [];
  for (let a = 0; a < confusion.length; a++) {
    for (let p = 0; p < confusion.length; p++) {
      if (a !== p && confusion[a][p] > 0) out.push({ actual: labels[a], predicted: labels[p], count: confusion[a][p] });
    }
  }
  return out.sort((x, y) => y.count - x.count).slice(0, limit);
}

export interface LayerChange {
  layer: 1 | 2 | 3;
  /** mean |Δw| */
  meanAbs: number;
  maxAbs: number;
  /** mean |Δw| relative to mean |w| before */
  relative: number;
}

/** How much every layer's weights moved between two snapshots. */
export function weightChange(prev: NetworkWeights, next: NetworkWeights): LayerChange[] {
  return ([1, 2, 3] as const).map((layer) => {
    const a = prev[`w${layer}` as 'w1'];
    const b = next[`w${layer}` as 'w1'];
    let sum = 0;
    let max = 0;
    let base = 0;
    for (let i = 0; i < a.length; i++) {
      const d = Math.abs(b[i] - a[i]);
      sum += d;
      if (d > max) max = d;
      base += Math.abs(a[i]);
    }
    return { layer, meanAbs: sum / Math.max(1, a.length), maxAbs: max, relative: base > 0 ? sum / base : 0 };
  });
}

export function copyWeights(w: NetworkWeights): NetworkWeights {
  return { w1: w.w1.slice(), b1: w.b1.slice(), w2: w.w2.slice(), b2: w.b2.slice(), w3: w.w3.slice(), b3: w.b3.slice() };
}
