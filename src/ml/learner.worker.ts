/// <reference lib="webworker" />
/**
 * Runs fineTune (see userModel.ts) in a background thread, so that learning from a correction
 * never freezes the canvas. The original training examples used for replay are downloaded once
 * and kept for the following corrections.
 */
import { datasetUrl, fetchDataset, toTrainingSamples } from './datasets';
import { NeuralNetwork } from './neuralNetwork';
import type { NetworkWeights } from './trainingClient';
import { setWeights } from './trainingClient';
import type { TrainingSample } from '../types';
import { Correction, fineTune, FineTuneResult } from './userModel';
import type { NetworkConfig } from '../types';

export interface LearnRequest {
  type: 'learn';
  config: NetworkConfig;
  weights: NetworkWeights;
  classes: string[];
  corrections: Correction[];
  newestId: string;
}

export type LearnResponse =
  | { type: 'learned'; weights: NetworkWeights; result: FineTuneResult }
  | { type: 'error'; message: string };

let replay: Promise<TrainingSample[]> | null = null;

function loadReplay(): Promise<TrainingSample[]> {
  if (!replay) {
    replay = fetchDataset(datasetUrl('symbols-replay')).then((ds) => toTrainingSamples(ds, 'replay'));
    replay.catch(() => (replay = null));
  }
  return replay;
}

self.onmessage = async (event: MessageEvent<LearnRequest>) => {
  const req = event.data;
  try {
    const net = new NeuralNetwork(req.config);
    setWeights(net, req.weights);
    const newest = req.corrections.find((c) => c.id === req.newestId);
    if (!newest) throw new Error('Correction not found');
    const result = fineTune(net, req.classes, req.corrections, newest, await loadReplay());
    const weights: NetworkWeights = {
      w1: net.w1.slice(),
      b1: net.b1.slice(),
      w2: net.w2.slice(),
      b2: net.b2.slice(),
      w3: net.w3.slice(),
      b3: net.b3.slice(),
    };
    self.postMessage({ type: 'learned', weights, result } satisfies LearnResponse);
  } catch (err) {
    self.postMessage({ type: 'error', message: err instanceof Error ? err.message : String(err) } satisfies LearnResponse);
  }
};
