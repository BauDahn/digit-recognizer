import { NetworkConfig } from '../types';
import { kvDelete, kvGet, kvSet } from './kvStore';
import type { LearnRequest, LearnResponse } from './learner.worker';
import { LoadedModel, loadBuiltinModel } from './modelFile';
import { NeuralNetwork } from './neuralNetwork';
import { getWeights, NetworkWeights, setWeights } from './trainingClient';
import { Correction, FineTuneResult } from './userModel';

/**
 * The network behind the math canvas: the shipped model (public/models/math.json) plus what this
 * person taught it. The lessons (corrections) and the resulting weights live in IndexedDB.
 */

const KEY = 'math-user-model';

interface StoredUserModel {
  /** Identifies the shipped model the lessons were applied on top of */
  signature: string;
  weights: NetworkWeights;
  corrections: Correction[];
}

export interface MathModelHandle {
  model: LoadedModel;
  corrections: Correction[];
  /** True when the weights include lessons from corrections */
  personalized: boolean;
}

function signatureOf(model: LoadedModel): string {
  const m = model.meta;
  return `${model.classes.length}:${m?.createdAt}:${m?.epochs}:${m?.testAccuracy}`;
}

function cloneNetwork(source: NeuralNetwork): NeuralNetwork {
  const copy = new NeuralNetwork(source.config);
  copy.copyWeightsFrom(source);
  return copy;
}

export async function loadMathModel(): Promise<MathModelHandle> {
  const base = await loadBuiltinModel('math');
  const model: LoadedModel = { ...base, network: cloneNetwork(base.network) };
  const stored = await kvGet<StoredUserModel>(KEY);
  if (stored && stored.signature === signatureOf(base)) {
    try {
      setWeights(model.network, stored.weights);
      return { model, corrections: stored.corrections, personalized: stored.corrections.length > 0 };
    } catch {
      await kvDelete(KEY);
    }
  }
  return { model, corrections: [], personalized: false };
}

export async function saveUserModel(model: LoadedModel, base: LoadedModel, corrections: Correction[]) {
  const weights = getWeights(model.network);
  const copy = {} as NetworkWeights;
  for (const k of Object.keys(weights) as Array<keyof NetworkWeights>) copy[k] = weights[k].slice();
  await kvSet(KEY, { signature: signatureOf(base), weights: copy, corrections } satisfies StoredUserModel);
}

export async function forgetUserModel() {
  await kvDelete(KEY);
}

let worker: Worker | null = null;

/** Teach the network the newest correction, in a background thread. Resolves with the new weights. */
export function learnCorrection(
  config: NetworkConfig,
  weights: NetworkWeights,
  classes: string[],
  corrections: Correction[],
  newestId: string
): Promise<{ weights: NetworkWeights; result: FineTuneResult }> {
  if (!worker) worker = new Worker(new URL('./learner.worker.ts', import.meta.url), { type: 'module' });
  const w = worker;
  return new Promise((resolve, reject) => {
    w.onmessage = (event: MessageEvent<LearnResponse>) => {
      const msg = event.data;
      if (msg.type === 'learned') resolve({ weights: msg.weights, result: msg.result });
      else reject(new Error(msg.message));
    };
    w.onerror = (e) => reject(new Error(e.message));
    const copy = {} as NetworkWeights;
    for (const k of Object.keys(weights) as Array<keyof NetworkWeights>) copy[k] = weights[k].slice();
    const request: LearnRequest = { type: 'learn', config, weights: copy, classes, corrections, newestId };
    w.postMessage(request);
  });
}
