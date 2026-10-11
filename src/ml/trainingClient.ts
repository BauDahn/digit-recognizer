import { NetworkConfig } from '../types';
import { IMG_PIXELS } from './imageOps';
import { NeuralNetwork } from './neuralNetwork';

/** Types and helpers shared by the main thread and src/ml/trainer.worker.ts */

export interface PackedSamples {
  labels: Uint8Array;
  pixels: Float32Array; // count * 784
}

export interface NetworkWeights {
  w1: Float32Array;
  b1: Float32Array;
  w2: Float32Array;
  b2: Float32Array;
  w3: Float32Array;
  b3: Float32Array;
}

export interface TrainRequest {
  type: 'train';
  config: NetworkConfig;
  initialWeights: NetworkWeights | null; // null = fresh He initialization
  train: PackedSamples;
  test: PackedSamples;
  epochs: number;
  augmentProbability: number; // 0 = no data augmentation
  /** How many times per epoch the worker sends the current weights (default 1: only at the end) */
  updatesPerEpoch?: number;
}

export interface EpochMessage {
  type: 'epoch';
  epoch: number;
  loss: number;
  trainAccuracy: number;
  testAccuracy: number;
  testLoss: number;
  confusionMatrix: number[][];
  weights: NetworkWeights;
  seconds: number;
}

/** Sent in the middle of an epoch so the lab can show the neurons being adjusted. */
export interface ProgressMessage {
  type: 'progress';
  epoch: number;
  /** how much of the epoch is done, 0..1 */
  fraction: number;
  loss: number;
  trainAccuracy: number;
  weights: NetworkWeights;
}

export type WorkerMessage = EpochMessage | ProgressMessage | { type: 'done' } | { type: 'error'; message: string };

/** Pack samples into two flat typed arrays (fast to copy to the worker). */
export function packSamples(samples: Array<{ label: number; pixels: Float32Array }>): PackedSamples {
  const labels = new Uint8Array(samples.length);
  const pixels = new Float32Array(samples.length * IMG_PIXELS);
  samples.forEach((s, i) => {
    labels[i] = s.label;
    pixels.set(s.pixels, i * IMG_PIXELS);
  });
  return { labels, pixels };
}

export function getWeights(net: NeuralNetwork): NetworkWeights {
  return { w1: net.w1, b1: net.b1, w2: net.w2, b2: net.b2, w3: net.w3, b3: net.b3 };
}

export function setWeights(net: NeuralNetwork, weights: NetworkWeights) {
  for (const key of ['w1', 'b1', 'w2', 'b2', 'w3', 'b3'] as const) net[key].set(weights[key]);
  net.isTrained = true;
}

export function createTrainerWorker(): Worker {
  return new Worker(new URL('./trainer.worker.ts', import.meta.url), { type: 'module' });
}
