/**
 * Trains the pre-trained models shipped in public/models/ with the exact same
 * NeuralNetwork class the browser uses:
 *   - digits.json   784-64-32-10   MNIST                       (DigitLab default architecture)
 *   - letters.json  784-128-64-47  EMNIST Balanced             (Alphanumeric Studio)
 *   - math.json     784-512-256-N  MNIST + EMNIST + HASYv2 + drawn symbols (Math canvas, `npm run build:symbols` first)
 *
 * Usage: npm run train:models [-- digits letters symbols]
 */
import { mkdirSync, readFileSync, writeFileSync } from 'node:fs';
import path from 'node:path';
import { DIGIT_CLASSES, EMNIST_BALANCED_CLASSES } from '../src/ml/classes';
import { decodeDataset } from '../src/ml/datasets';
import { MATH_CLASSES } from '../src/ml/symbols';
import { augment28, IMG_PIXELS, seededRandom } from '../src/ml/imageOps';
import { toModelFile } from '../src/ml/modelFile';
import { NeuralNetwork } from '../src/ml/neuralNetwork';
import { ParallelTrainer } from './lib/parallelTrainer';
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
  trainedOn: string,
  { lrDecay = 0.75, checkpoints = false }: { lrDecay?: number; checkpoints?: boolean } = {}
) {
  const net = new NeuralNetwork({ ...config, outputSize: classes.length });
  const baseLr = net.config.learningRate;
  console.log(`\n[${name}] ${trainSet.length} train / ${testSet.length} test samples`);
  for (let epoch = 1; epoch <= epochs; epoch++) {
    // Step decay: keep the learning rate for the first epochs, then shrink it
    net.config.learningRate = baseLr * Math.pow(lrDecay, Math.max(0, epoch - 3));
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
    if (checkpoints) {
      const meta = { trainedOn, testAccuracy: Math.round(test.accuracy * 10000) / 10000, epochs: epoch };
      writeFileSync(path.join(OUT_DIR, `${name}.json`), JSON.stringify(toModelFile(name, classes, net, meta)));
    }
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

/**
 * The big math model: trains on everything in .cache/symbols with balanced classes,
 * spread over several CPU cores (see scripts/lib/parallelTrainer.ts).
 */
async function trainMath() {
  const load = (name: string) =>
    decodeDataset(new Uint8Array(readFileSync(path.resolve(import.meta.dirname, `../.cache/symbols/${name}.bin`))));
  const limit = Number(process.env.MATH_SUBSET ?? 0);
  const epochs = Number(process.env.MATH_EPOCHS ?? 24);
  const baseLr = Number(process.env.MATH_LR ?? 0.05);
  const train = load('train');
  const testSet = toSamples({ labels: train.labels.slice(0, 0), images: new Uint8Array(0) });
  const test = load('test');
  testSet.push(...toSamples({ labels: test.labels, images: test.images }));
  const count = limit > 0 ? Math.min(limit, train.count) : train.count;

  // Classes with few images get repeated (every copy is augmented differently) so that
  // rare symbols like '∄' are not drowned by the 4,600 images of each digit.
  const byClass = new Map<number, number[]>();
  for (let i = 0; i < count; i++) byClass.set(train.labels[i], [...(byClass.get(train.labels[i]) ?? []), i]);
  const MIN_PER_CLASS = 900;
  const pool: number[] = [];
  for (const idx of byClass.values()) {
    pool.push(...idx);
    for (let i = idx.length; i < MIN_PER_CLASS && limit === 0; i++) pool.push(idx[i % idx.length]);
  }

  const net = new NeuralNetwork({
    hiddenSize1: 512,
    hiddenSize2: 256,
    outputSize: MATH_CLASSES.length,
    learningRate: baseLr,
    momentum: 0.9,
    batchSize: 32,
  });
  const trainer = new ParallelTrainer(net, train.images, train.labels);
  const trainedOn = 'MNIST + EMNIST ByClass + HASYv2 + drawn symbols';
  console.log(`\n[math] ${pool.length} train samples (${count} distinct) / ${testSet.length} test, ${trainer.threads} threads`);

  for (let epoch = 1; epoch <= epochs; epoch++) {
    const lr = baseLr * Math.pow(Number(process.env.MATH_DECAY ?? 0.92), Math.max(0, epoch - 2));
    for (let i = pool.length - 1; i > 0; i--) {
      const j = Math.floor(rand() * (i + 1));
      [pool[i], pool[j]] = [pool[j], pool[i]];
    }
    const t0 = Date.now();
    const { loss, accuracy } = await trainer.trainEpoch(Int32Array.from(pool), lr, 0.7, Number(process.env.MATH_ROUNDS ?? 20), epoch);
    const result = net.evaluate(testSet.slice(0, 4000)); // quick estimate; the full test set is used at the end
    console.log(
      `  epoch ${epoch}/${epochs}  lr ${lr.toFixed(4)}  loss ${loss.toFixed(3)}  train ${(accuracy * 100).toFixed(1)}%  ` +
        `test ${(result.accuracy * 100).toFixed(1)}%  (${((Date.now() - t0) / 1000).toFixed(1)}s)`
    );
    const meta = {
      trainedOn,
      testAccuracy: Math.round(result.accuracy * 10000) / 10000,
      epochs: epoch,
      createdAt: new Date().toISOString().slice(0, 10),
    };
    writeFileSync(path.join(OUT_DIR, 'math.json'), JSON.stringify(toModelFile('math', MATH_CLASSES, net, meta)));
  }
  await trainer.close();
  const final = net.evaluate(testSet);
  console.log(`[math] full test set (${testSet.length} images): ${(final.accuracy * 100).toFixed(2)}%`);
  writeFileSync(
    path.join(OUT_DIR, 'math.json'),
    JSON.stringify(
      toModelFile('math', MATH_CLASSES, net, {
        trainedOn,
        testAccuracy: Math.round(final.accuracy * 10000) / 10000,
        epochs,
        createdAt: new Date().toISOString().slice(0, 10),
      })
    )
  );
}

const requested = new Set(process.argv.slice(2));
const wants = (name: string) => requested.size === 0 || requested.has(name);
mkdirSync(OUT_DIR, { recursive: true });

if (wants('digits')) {
  train(
    'digits',
    DIGIT_CLASSES,
    { hiddenSize1: 64, hiddenSize2: 32 },
    toSamples(await loadMnist('train')),
    toSamples(await loadMnist('test')),
    8,
    0.3,
    'MNIST (60,000 train / 10,000 test)'
  );
}

if (requested.has('math')) await trainMath();

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
