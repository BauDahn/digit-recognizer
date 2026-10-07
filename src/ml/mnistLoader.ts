import { TrainingSample } from '../types';
import { datasetUrl, fetchDataset, imageAt, RawDataset, toTrainingSamples } from './datasets';

/**
 * Real MNIST, served from public/datasets (built by `npm run build:datasets`):
 * 8,000 training images (800 per digit) and 2,000 test images (200 per digit),
 * taken from the official MNIST train and test splits.
 */

export interface MnistSampleInfo {
  id: string;
  label: number;
  pixels: Float32Array;
}

let trainPromise: Promise<RawDataset> | null = null;
let testPromise: Promise<RawDataset> | null = null;

function loadRaw(split: 'train' | 'test'): Promise<RawDataset> {
  if (split === 'train') {
    trainPromise ??= fetchDataset(datasetUrl('mnist-train')).catch((e) => {
      trainPromise = null;
      throw e;
    });
    return trainPromise;
  }
  testPromise ??= fetchDataset(datasetUrl('mnist-test')).catch((e) => {
    testPromise = null;
    throw e;
  });
  return testPromise;
}

export async function getRealMnistSplits(): Promise<{ trainSet: TrainingSample[]; testSet: TrainingSample[] }> {
  const [train, test] = await Promise.all([loadRaw('train'), loadRaw('test')]);
  return { trainSet: toTrainingSamples(train, 'mnist-train'), testSet: toTrainingSamples(test, 'mnist-test') };
}

/** A few samples of each digit from the test split, for the gallery. */
export async function loadMnistGallerySamples(perDigit = 30): Promise<MnistSampleInfo[]> {
  const test = await loadRaw('test');
  const counts = new Array(10).fill(0);
  const out: MnistSampleInfo[] = [];
  for (let i = 0; i < test.count; i++) {
    const label = test.labels[i];
    if (counts[label] >= perDigit) continue;
    counts[label]++;
    out.push({ id: `mnist-test-${i}`, label, pixels: imageAt(test, i) });
  }
  return out;
}
