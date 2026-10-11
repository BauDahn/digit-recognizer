/** Types and constants shared by the parallel trainer and its worker threads. */
import { NetworkConfig } from '../../src/types';

export const PARAM_NAMES = ['w1', 'b1', 'w2', 'b2', 'w3', 'b3', 'vw1', 'vb1', 'vw2', 'vb2', 'vw3', 'vb3'] as const;
export type ParamName = (typeof PARAM_NAMES)[number];

export interface WorkerInit {
  id: number;
  images: SharedArrayBuffer;
  labels: SharedArrayBuffer;
  config: NetworkConfig;
}

export interface RoundRequest {
  params: Record<ParamName, Float32Array>;
  learningRate: number;
  indices: Int32Array;
  augmentProb: number;
  seed: number;
}

export interface RoundResult {
  params: Record<ParamName, Float32Array>;
  loss: number;
  accuracy: number;
  count: number;
}
