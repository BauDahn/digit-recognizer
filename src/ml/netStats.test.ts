import { describe, expect, it } from 'vitest';
import { NeuralNetwork } from './neuralNetwork';
import { classReport, copyWeights, summarizeNetwork, topConfusions, weightChange } from './netStats';
import { getWeights } from './trainingClient';

describe('summarizeNetwork', () => {
  it('counts parameters layer by layer', () => {
    const net = new NeuralNetwork({ hiddenSize1: 64, hiddenSize2: 32, outputSize: 10 });
    const s = summarizeNetwork(net);
    expect(s.layers.map((l) => l.params)).toEqual([784 * 64 + 64, 64 * 32 + 32, 32 * 10 + 10]);
    expect(s.totalParams).toBe(52650);
    expect(s.bytes).toBe(52650 * 4);
    expect(s.layers[0].histogram.reduce((a, b) => a + b, 0)).toBe(784 * 64);
  });

  it('shows He initialization: zero mean, std close to sqrt(2 / fan-in)', () => {
    const net = new NeuralNetwork({ hiddenSize1: 128, hiddenSize2: 64, outputSize: 10 });
    const [l1] = summarizeNetwork(net).layers;
    expect(Math.abs(l1.mean)).toBeLessThan(0.01);
    expect(l1.std).toBeCloseTo(Math.sqrt(2 / 784), 2);
  });
});

describe('classReport', () => {
  const confusion = [
    [8, 2, 0],
    [1, 9, 0],
    [0, 0, 10],
  ];
  it('computes precision, recall and F1', () => {
    const r = classReport(confusion, ['a', 'b', 'c']);
    expect(r.accuracy).toBeCloseTo(27 / 30);
    expect(r.perClass[0].precision).toBeCloseTo(8 / 9);
    expect(r.perClass[0].recall).toBeCloseTo(8 / 10);
    expect(r.perClass[2].f1).toBe(1);
    expect(r.total).toBe(30);
  });

  it('lists the most frequent mistakes', () => {
    expect(topConfusions(confusion, ['a', 'b', 'c'], 2)).toEqual([
      { actual: 'a', predicted: 'b', count: 2 },
      { actual: 'b', predicted: 'a', count: 1 },
    ]);
  });
});

describe('weightChange', () => {
  it('measures how far the weights moved during training', () => {
    const net = new NeuralNetwork({ hiddenSize1: 8, hiddenSize2: 4, outputSize: 3 });
    const before = copyWeights(getWeights(net));
    net.w1[0] += 0.5;
    const [l1, l2] = weightChange(before, getWeights(net));
    expect(l1.maxAbs).toBeCloseTo(0.5);
    expect(l1.meanAbs).toBeCloseTo(0.5 / net.w1.length);
    expect(l2.meanAbs).toBe(0);
  });
});
