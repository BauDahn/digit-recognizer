/**
 * Trains the pre-trained models shipped in public/models/ with the exact same
 * NeuralNetwork class the browser uses:
 *   - digits.json   784-64-32-10   MNIST                       (DigitLab default architecture)
 *   - letters.json  784-128-64-47  EMNIST Balanced             (Alphanumeric Studio)
 *   - symbols.json  784-128-64-17  MNIST + operators + 'x'     (Math canvas)
 *
 * Usage: npm run train:models [-- digits letters symbols]
 */
import { mkdirSync, writeFileSync } from 'node:fs';
import path from 'node:path';
import { DIGIT_CLASSES, EMNIST_BALANCED_CLASSES, MATH_SYMBOL_CLASSES } from '../src/ml/classes';
import { augment28, IMG_PIXELS, seededRandom } from '../src/ml/imageOps';
import { toModelFile } from '../src/ml/modelFile';
import { NeuralNetwork } from '../src/ml/neuralNetwork';
import { generateOperatorImage, OperatorSymbol } from '../src/ml/syntheticSymbols';
import { NetworkConfig, TrainingSample } from '../src/types';
import { IdxDataset, loadEmnistBalanced, loadMnist, stratifiedSubset } from './lib/idx';

const OUT_DIR = path.resolve(import.meta.dirname, '../public/models');
const rand = seededRandom(42);

function toSamples(ds: IdxDataset, labelMap: (label: number) => number = (l) => l): TrainingSample[] {
  const out: TrainingSample[] = [];
  for (let i = 0; i < ds.labels.length; i++) {
    const pixels = new Float32Array(IMG_PIXELS);
    for (let p = 0; p < IMG_PIXELS; p++) pixels[p] = ds.images[i * IMG_PIXELS + p] / 255;
    out.push({ id: String(i), label: labelMap(ds.labels[i]), pixels });
  }
  return out;
}

function train(
  name: string,
  classes: string[],
  config: Partial<NetworkConfig>,
  trainSet: TrainingSample[],
  testSet: TrainingSample[],
  epochs: number,
  augmentProb: number,
  trainedOn: string
) {
  const net = new NeuralNetwork({ ...config, outputSize: classes.length });
  const baseLr = net.config.learningRate;
  console.log(`\n[${name}] ${trainSet.length} train / ${testSet.length} test samples`);
  for (let epoch = 1; epoch <= epochs; epoch++) {
    // Step decay: keep the learning rate for the first epochs, then shrink it
    net.config.learningRate = baseLr * Math.pow(0.75, Math.max(0, epoch - 3));
    const epochSet = trainSet.map((s) =>
      rand() < augmentProb ? { ...s, pixels: augment28(s.pixels, {}, rand) } : s
    );
    const t0 = Date.now();
    const { loss, accuracy } = net.trainEpoch(epochSet);
    const test = net.evaluate(testSet);
    console.log(
      `  epoch ${epoch}/${epochs}  loss ${loss.toFixed(3)}  train ${(accuracy * 100).toFixed(1)}%  ` +
        `test ${(test.accuracy * 100).toFixed(1)}%  (${((Date.now() - t0) / 1000).toFixed(1)}s)`
    );
  }
  net.config.learningRate = baseLr;
  const testAccuracy = net.evaluate(testSet).accuracy;
  const file = toModelFile(name, classes, net, {
    trainedOn,
    testAccuracy: Math.round(testAccuracy * 10000) / 10000,
    epochs,
    createdAt: new Date().toISOString().slice(0, 10),
  });
  writeFileSync(path.join(OUT_DIR, `${name}.json`), JSON.stringify(file));
  console.log(`[${name}] saved, test accuracy ${(testAccuracy * 100).toFixed(2)}%`);
}

function operatorSamples(perClass: number): TrainingSample[] {
  const ops: OperatorSymbol[] = ['+', '-', '/', '=', '(', ')', 'x'];
  const out: TrainingSample[] = [];
  for (const op of ops) {
    const label = MATH_SYMBOL_CLASSES.indexOf(op);
    for (let i = 0; i < perClass; i++) {
      out.push({ id: `${op}-${i}`, label, pixels: generateOperatorImage(op, rand) });
    }
  }
  return out;
}

const requested = new Set(process.argv.slice(2));
const wants = (name: string) => requested.size === 0 || requested.has(name);
mkdirSync(OUT_DIR, { recursive: true });

if (wants('digits') || wants('symbols')) {
  const mnistTrain = await loadMnist('train');
  const mnistTest = await loadMnist('test');

  if (wants('digits')) {
    train(
      'digits',
      DIGIT_CLASSES,
      { hiddenSize1: 64, hiddenSize2: 32 },
      toSamples(mnistTrain),
      toSamples(mnistTest),
      8,
      0.3,
      'MNIST (60,000 train / 10,000 test)'
    );
  }

  if (wants('symbols')) {
    const emnistTrain = await loadEmnistBalanced('train');
    const emnistTest = await loadEmnistBalanced('test');
    const emnistX = EMNIST_BALANCED_CLASSES.indexOf('X');
    const xLabel = MATH_SYMBOL_CLASSES.indexOf('x');
    const onlyX = (ds: IdxDataset, n: number) => {
      const subset = stratifiedSubset(ds, n, EMNIST_BALANCED_CLASSES.length);
      return toSamples(subset)
        .filter((s) => s.label === emnistX)
        .map((s) => ({ ...s, label: xLabel }));
    };

    const trainSet = [
      ...toSamples(stratifiedSubset(mnistTrain, 2500, 10)),
      ...operatorSamples(2000),
      ...onlyX(emnistTrain, 1500),
    ];
    const testSet = [
      ...toSamples(stratifiedSubset(mnistTest, 300, 10)),
      ...operatorSamples(300),
      ...onlyX(emnistTest, 300),
    ];
    train(
      'symbols',
      MATH_SYMBOL_CLASSES,
      { hiddenSize1: 128, hiddenSize2: 64 },
      trainSet,
      testSet,
      8,
      0.5,
      'MNIST + synthetic operators + "x" from EMNIST'
    );
  }
}

if (wants('letters')) {
  train(
    'letters',
    EMNIST_BALANCED_CLASSES,
    { hiddenSize1: 128, hiddenSize2: 64 },
    toSamples(stratifiedSubset(await loadEmnistBalanced('train'), 1500, EMNIST_BALANCED_CLASSES.length)),
    toSamples(await loadEmnistBalanced('test')),
    10,
    0.3,
    'EMNIST Balanced (70,500 train / 18,800 test)'
  );
}
