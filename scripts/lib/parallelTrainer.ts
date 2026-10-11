/**
 * Trains one network on several CPU cores with "local SGD": every worker trains a copy of the
 * network on its share of the data for a short round, then the copies (weights and momentum)
 * are averaged and the next round starts from the average.
 */
import os from 'node:os';
import { Worker } from 'node:worker_threads';
import { NeuralNetwork } from '../../src/ml/neuralNetwork';
import { NetworkConfig } from '../../src/types';
import { ParamName, PARAM_NAMES, RoundRequest, RoundResult, WorkerInit } from './trainShared';

export class ParallelTrainer {
  private workers: Worker[] = [];

  constructor(
    private net: NeuralNetwork,
    images: Uint8Array,
    labels: Uint8Array,
    threads = Math.max(2, Math.min(14, os.cpus().length - 2))
  ) {
    const sharedImages = new SharedArrayBuffer(images.length);
    new Uint8Array(sharedImages).set(images);
    const sharedLabels = new SharedArrayBuffer(labels.length);
    new Uint8Array(sharedLabels).set(labels);
    const config: NetworkConfig = { ...net.config };
    for (let id = 0; id < threads; id++) {
      const init: WorkerInit = { id, images: sharedImages, labels: sharedLabels, config };
      this.workers.push(
        new Worker(new URL('./trainWorker.ts', import.meta.url), { workerData: init, execArgv: ['--import', 'tsx'] })
      );
    }
  }

  get threads() {
    return this.workers.length;
  }

  private runRound(worker: Worker, request: RoundRequest): Promise<RoundResult> {
    return new Promise((resolve, reject) => {
      const onError = (err: Error) => reject(err);
      worker.once('message', (result: RoundResult) => {
        worker.off('error', onError);
        resolve(result);
      });
      worker.once('error', onError);
      worker.postMessage(request);
    });
  }

  /** One epoch over `indices` (already shuffled), averaging the workers `rounds` times. */
  async trainEpoch(indices: Int32Array, learningRate: number, augmentProb: number, rounds: number, seed: number) {
    const k = this.workers.length;
    const perRound = Math.floor(indices.length / rounds);
    let loss = 0;
    let correct = 0;
    let total = 0;

    for (let r = 0; r < rounds; r++) {
      const roundIndices = indices.subarray(r * perRound, (r + 1) * perRound);
      const shard = Math.floor(roundIndices.length / k);
      const results = await Promise.all(
        this.workers.map((worker, w) => {
          const params = {} as Record<ParamName, Float32Array>;
          for (const name of PARAM_NAMES) params[name] = this.net[name].slice();
          const request: RoundRequest = {
            params,
            learningRate,
            indices: roundIndices.slice(w * shard, (w + 1) * shard),
            augmentProb,
            seed: seed * 7919 + r * 131 + w,
          };
          return this.runRound(worker, request);
        })
      );

      for (const name of PARAM_NAMES) {
        const target = this.net[name];
        target.fill(0);
        for (const res of results) {
          const p = res.params[name];
          for (let i = 0; i < target.length; i++) target[i] += p[i];
        }
        for (let i = 0; i < target.length; i++) target[i] /= results.length;
      }
      for (const res of results) {
        loss += res.loss * res.count;
        correct += res.accuracy * res.count;
        total += res.count;
      }
    }
    this.net.isTrained = true;
    return { loss: loss / total, accuracy: correct / total };
  }

  async close() {
    await Promise.all(this.workers.map((w) => w.terminate()));
  }
}
