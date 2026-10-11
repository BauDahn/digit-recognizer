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
import type { EpochMessage, PackedSamples, ProgressMessage, TrainRequest, WorkerMessage } from './trainingClient';

function unpack(p: PackedSamples) {
  return Array.from(p.labels, (label, i) => ({
    id: String(i),
    label,
    pixels: p.pixels.subarray(i * IMG_PIXELS, (i + 1) * IMG_PIXELS),
  }));
}

function snapshot(net: NeuralNetwork) {
  return { w1: net.w1.slice(), b1: net.b1.slice(), w2: net.w2.slice(), b2: net.b2.slice(), w3: net.w3.slice(), b3: net.b3.slice() };
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
      // The epoch is trained in chunks (one pass over every sample in total) so that the lab can
      // show the weights while they change.
      const chunks = Math.max(1, req.updatesPerEpoch ?? 1);
      const order = epochSet.slice();
      for (let i = order.length - 1; i > 0; i--) {
        const j = Math.floor(Math.random() * (i + 1));
        [order[i], order[j]] = [order[j], order[i]];
      }
      const size = Math.ceil(order.length / chunks);
      let lossSum = 0;
      let correct = 0;
      let seen = 0;
      for (let c = 0; c < chunks; c++) {
        const part = order.slice(c * size, (c + 1) * size);
        if (part.length === 0) continue;
        const r = net.trainEpoch(part);
        lossSum += r.loss * part.length;
        correct += r.accuracy * part.length;
        seen += part.length;
        if (c < chunks - 1) {
          self.postMessage({
            type: 'progress',
            epoch,
            fraction: seen / order.length,
            loss: lossSum / seen,
            trainAccuracy: correct / seen,
            weights: snapshot(net),
          } satisfies ProgressMessage);
        }
      }
      const loss = lossSum / Math.max(1, seen);
      const accuracy = correct / Math.max(1, seen);
      const test = net.evaluate(testSet);
      const msg: EpochMessage = {
        type: 'epoch',
        epoch,
        loss,
        trainAccuracy: accuracy,
        testAccuracy: test.accuracy,
        testLoss: test.loss,
        confusionMatrix: test.confusionMatrix,
        weights: snapshot(net),
        seconds: (performance.now() - t0) / 1000,
      };
      self.postMessage(msg);
    }
    self.postMessage({ type: 'done' } satisfies WorkerMessage);
  } catch (err) {
    self.postMessage({ type: 'error', message: err instanceof Error ? err.message : String(err) } satisfies WorkerMessage);
  }
};
