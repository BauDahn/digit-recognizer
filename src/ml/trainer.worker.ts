/// <reference lib="webworker" />
/**
 * Trains a NeuralNetwork in a background thread so the page stays responsive.
 *
 * Main thread -> worker:  TrainRequest (once; to stop, the main thread terminates the worker)
 * Worker -> main thread:  one EpochMessage per epoch with metrics and the current weights,
 *                         then a DoneMessage (or an ErrorMessage).
 */
import { augment28, IMG_PIXELS } from './imageOps';
import { NeuralNetwork } from './neuralNetwork';
import { setWeights } from './trainingClient';
import type { EpochMessage, PackedSamples, TrainRequest, WorkerMessage } from './trainingClient';

function unpack(p: PackedSamples) {
  return Array.from(p.labels, (label, i) => ({
    id: String(i),
    label,
    pixels: p.pixels.subarray(i * IMG_PIXELS, (i + 1) * IMG_PIXELS),
  }));
}

self.onmessage = (event: MessageEvent<TrainRequest>) => {
  const req = event.data;
  try {
    const net = new NeuralNetwork(req.config);
    if (req.initialWeights) {
      setWeights(net, req.initialWeights);
    }
    const trainSet = unpack(req.train);
    const testSet = unpack(req.test);

    for (let epoch = 1; epoch <= req.epochs; epoch++) {
      const t0 = performance.now();
      const epochSet =
        req.augmentProbability > 0
          ? trainSet.map((s) => (Math.random() < req.augmentProbability ? { ...s, pixels: augment28(s.pixels) } : s))
          : trainSet;
      const { loss, accuracy } = net.trainEpoch(epochSet);
      const test = net.evaluate(testSet);
      const msg: EpochMessage = {
        type: 'epoch',
        epoch,
        loss,
        trainAccuracy: accuracy,
        testAccuracy: test.accuracy,
        testLoss: test.loss,
        confusionMatrix: test.confusionMatrix,
        weights: {
          w1: net.w1.slice(),
          b1: net.b1.slice(),
          w2: net.w2.slice(),
          b2: net.b2.slice(),
          w3: net.w3.slice(),
          b3: net.b3.slice(),
        },
        seconds: (performance.now() - t0) / 1000,
      };
      self.postMessage(msg);
    }
    self.postMessage({ type: 'done' } satisfies WorkerMessage);
  } catch (err) {
    self.postMessage({ type: 'error', message: err instanceof Error ? err.message : String(err) } satisfies WorkerMessage);
  }
};
