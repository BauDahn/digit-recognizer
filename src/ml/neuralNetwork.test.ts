import { describe, expect, it } from 'vitest';
import { NeuralNetwork } from './neuralNetwork';
import { seededRandom } from './imageOps';
import { TrainingSample } from '../types';

/** Tiny separable problem: class k lights up the k-th band of rows. */
function bandsDataset(classes: number, perClass: number): TrainingSample[] {
  const rand = seededRandom(7);
  const out: TrainingSample[] = [];
  const band = Math.floor(28 / classes);
  for (let label = 0; label < classes; label++) {
    for (let n = 0; n < perClass; n++) {
      const pixels = new Float32Array(784);
      for (let y = label * band; y < (label + 1) * band; y++) {
        for (let x = 0; x < 28; x++) pixels[y * 28 + x] = 0.6 + 0.4 * rand();
      }
      out.push({ id: `${label}-${n}`, label, pixels });
    }
  }
  return out;
}

describe('NeuralNetwork', () => {
  it('outputs a probability distribution of size outputSize', () => {
    const net = new NeuralNetwork({ hiddenSize1: 16, hiddenSize2: 8, outputSize: 5 });
    const { probabilities } = net.forward(new Float32Array(784).fill(0.5));
    expect(probabilities).toHaveLength(5);
    expect(probabilities.reduce((a, b) => a + b, 0)).toBeCloseTo(1, 5);
    expect(Math.min(...probabilities)).toBeGreaterThan(0);
  });

  it('learns a simple problem: loss goes down and accuracy goes up', () => {
    const data = bandsDataset(4, 20);
    const net = new NeuralNetwork({ hiddenSize1: 16, hiddenSize2: 8, outputSize: 4, learningRate: 0.05 });
    const before = net.evaluate(data);
    let last = { loss: Infinity, accuracy: 0 };
    for (let i = 0; i < 10; i++) last = net.trainEpoch(data);
    const after = net.evaluate(data);
    expect(after.loss).toBeLessThan(before.loss);
    expect(after.accuracy).toBeGreaterThan(0.95);
    expect(last.accuracy).toBeGreaterThan(0.9);
    expect(after.confusionMatrix).toHaveLength(4);
    expect(after.confusionMatrix.flat().reduce((a, b) => a + b, 0)).toBe(data.length);
  });

  it('serializes and restores the exact same predictions', () => {
    const net = new NeuralNetwork({ hiddenSize1: 12, hiddenSize2: 6, outputSize: 3 });
    const input = new Float32Array(784).map((_, i) => (i % 7) / 7);
    const restored = NeuralNetwork.deserialize(JSON.parse(net.exportWeights()));
    expect(restored.config).toEqual(net.config);
    expect(Array.from(restored.forward(input).probabilities)).toEqual(Array.from(net.forward(input).probabilities));
  });

  it('refuses to import weights of a different architecture', () => {
    const small = new NeuralNetwork({ hiddenSize1: 8, hiddenSize2: 4 });
    const big = new NeuralNetwork({ hiddenSize1: 16, hiddenSize2: 4 });
    expect(() => big.importWeights(small.exportWeights())).toThrow(/Architecture mismatch/);
  });
});
