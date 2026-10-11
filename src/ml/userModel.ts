import { TrainingSample } from '../types';
import { augment28, IMG_PIXELS } from './imageOps';
import { NeuralNetwork } from './neuralNetwork';

/**
 * Teaching the network from corrections.
 *
 * When the user says "that symbol was a ∂, not a δ", the pixels the network saw become a new
 * labeled example. Training only on it would make the network forget everything else, so each
 * update mixes:
 *   - the new example (and slightly distorted copies of it, so the lesson generalizes),
 *   - the earlier corrections, so old lessons are not unlearned,
 *   - a random sample of the original training data ("replay"), which anchors the rest of the
 *     network where it was.
 * A few epochs of gentle SGD later, the network reads the symbol the way the user said.
 */

export interface Correction {
  id: string;
  /** The class the user chose (a character of the model's class list) */
  char: string;
  /** The 28x28 image the network saw, 0..255 */
  pixels: Uint8Array;
  createdAt: number;
}

export interface FineTuneOptions {
  learningRate?: number;
  epochsPerRound?: number;
  maxRounds?: number;
  /** copies (augmented) of the newest correction per round */
  newCopies?: number;
  /** copies (augmented) of each older correction per round */
  oldCopies?: number;
  /** size of the replay sample per round */
  replayCount?: number;
  /** stop when the newest example is read correctly with at least this probability */
  targetConfidence?: number;
}

export interface FineTuneResult {
  rounds: number;
  before: { char: string; confidence: number; probabilityOfCorrect: number };
  after: { char: string; confidence: number; probabilityOfCorrect: number };
  learned: boolean;
}

export function toFloatPixels(bytes: Uint8Array): Float32Array {
  const out = new Float32Array(IMG_PIXELS);
  for (let i = 0; i < IMG_PIXELS; i++) out[i] = bytes[i] / 255;
  return out;
}

export function toBytePixels(pixels: Float32Array): Uint8Array {
  return Uint8Array.from(pixels, (v) => Math.round(Math.max(0, Math.min(1, v)) * 255));
}

function read(network: NeuralNetwork, classes: string[], pixels: Float32Array, label: number) {
  const { probabilities } = network.forward(pixels);
  let best = 0;
  for (let i = 1; i < probabilities.length; i++) if (probabilities[i] > probabilities[best]) best = i;
  return { char: classes[best], confidence: probabilities[best], probabilityOfCorrect: probabilities[label] };
}

function shuffle<T>(items: T[], random: () => number): T[] {
  for (let i = items.length - 1; i > 0; i--) {
    const j = Math.floor(random() * (i + 1));
    [items[i], items[j]] = [items[j], items[i]];
  }
  return items;
}

/**
 * Update `network` in place so that it reads `newest` as `newest.char`.
 * `corrections` is every correction made so far, newest included.
 */
export function fineTune(
  network: NeuralNetwork,
  classes: string[],
  corrections: Correction[],
  newest: Correction,
  replay: TrainingSample[],
  options: FineTuneOptions = {},
  random: () => number = Math.random
): FineTuneResult {
  const {
    learningRate = 0.01,
    epochsPerRound = 2,
    maxRounds = 6,
    newCopies = 10,
    oldCopies = 3,
    replayCount = 1000,
    targetConfidence = 0.7,
  } = options;

  const label = classes.indexOf(newest.char);
  if (label < 0) throw new Error(`"${newest.char}" is not one of the symbols this network knows`);
  const newestPixels = toFloatPixels(newest.pixels);
  const before = read(network, classes, newestPixels, label);

  const original = { ...network.config };
  let rounds = 0;
  let after = before;
  while (rounds < maxRounds) {
    rounds++;
    const samples: TrainingSample[] = [];
    for (let i = 0; i < newCopies; i++) {
      samples.push({ id: `${newest.id}-${i}`, label, pixels: i === 0 ? newestPixels : augment28(newestPixels, {}, random) });
    }
    // the most recent older corrections, at most ~150 distorted copies in total
    const older = corrections.filter((c) => c.id !== newest.id).slice(-Math.max(1, Math.floor(150 / oldCopies)));
    for (const c of older) {
      const pixels = toFloatPixels(c.pixels);
      for (let i = 0; i < oldCopies; i++) {
        samples.push({ id: `${c.id}-${i}`, label: classes.indexOf(c.char), pixels: i === 0 ? pixels : augment28(pixels, {}, random) });
      }
    }
    const picked = shuffle([...replay], random).slice(0, replayCount);
    samples.push(...picked);

    // Gentle SGD, a little stronger when the first round was not enough. Fresh momentum.
    network.config.learningRate = learningRate * (1 + 0.6 * (rounds - 1));
    network.config.momentum = 0.9;
    network.config.batchSize = 16;
    for (const v of [network.vw1, network.vb1, network.vw2, network.vb2, network.vw3, network.vb3]) v.fill(0);
    for (let e = 0; e < epochsPerRound; e++) network.trainEpoch(shuffle(samples, random));

    after = read(network, classes, newestPixels, label);
    if (after.char === newest.char && after.confidence >= targetConfidence) break;
  }
  network.config.learningRate = original.learningRate;
  network.config.momentum = original.momentum;
  network.config.batchSize = original.batchSize;

  return { rounds, before, after, learned: after.char === newest.char };
}
