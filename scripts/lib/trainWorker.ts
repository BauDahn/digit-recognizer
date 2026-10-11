/**
 * Worker thread of the parallel trainer (see parallelTrainer.ts): trains its own copy of the
 * network on a shard of the data and sends the weights back for averaging.
 */
import { parentPort, workerData } from 'node:worker_threads';
import { augment28, IMG_PIXELS, seededRandom } from '../../src/ml/imageOps';
import { NeuralNetwork } from '../../src/ml/neuralNetwork';
import { TrainingSample } from '../../src/types';
import { ParamName, PARAM_NAMES, RoundRequest, RoundResult, WorkerInit } from './trainShared';

const init = workerData as WorkerInit;
const images = new Uint8Array(init.images);
const labels = new Uint8Array(init.labels);
const net = new NeuralNetwork(init.config);

parentPort!.on('message', (req: RoundRequest) => {
  for (const name of PARAM_NAMES) net[name].set(req.params[name]);
  net.config.learningRate = req.learningRate;

  const rand = seededRandom(req.seed);
  const samples: TrainingSample[] = [];
  for (const idx of req.indices) {
    let pixels: Float32Array = new Float32Array(IMG_PIXELS);
    const offset = idx * IMG_PIXELS;
    for (let p = 0; p < IMG_PIXELS; p++) pixels[p] = images[offset + p] / 255;
    if (rand() < req.augmentProb) pixels = augment28(pixels, {}, rand);
    samples.push({ id: String(idx), label: labels[idx], pixels });
  }

  const { loss, accuracy } = net.trainEpoch(samples);
  const params = {} as Record<ParamName, Float32Array>;
  for (const name of PARAM_NAMES) params[name] = net[name].slice();
  const result: RoundResult = { params, loss, accuracy, count: samples.length };
  parentPort!.postMessage(
    result,
    PARAM_NAMES.map((n) => params[n].buffer as ArrayBuffer)
  );
});
