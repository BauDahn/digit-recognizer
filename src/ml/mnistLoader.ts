import rawMnistData from '../data/mnist_subset.json';
import { TrainingSample } from '../types';

/**
 * Decodes base64 string into Float32Array of 784 normalized [0, 1] pixels
 */
function decodeBase64ToFloat32Array(b64: string): Float32Array {
  const binaryString = atob(b64);
  const len = binaryString.length;
  const pixels = new Float32Array(len);
  for (let i = 0; i < len; i++) {
    pixels[i] = binaryString.charCodeAt(i) / 255.0;
  }
  return pixels;
}

export interface MnistSampleInfo {
  id: string;
  label: number;
  pixels: Float32Array;
}

let cachedMnistSamples: MnistSampleInfo[] | null = null;

export function loadRealMnistSamples(): MnistSampleInfo[] {
  if (cachedMnistSamples) return cachedMnistSamples;

  cachedMnistSamples = (rawMnistData as Array<{ id: string; label: number; b64: string }>).map(
    (item) => ({
      id: item.id,
      label: item.label,
      pixels: decodeBase64ToFloat32Array(item.b64),
    })
  );

  return cachedMnistSamples;
}

/**
 * Partition real MNIST into training set and test set
 */
export function getRealMnistSplits(trainRatio = 0.8): {
  trainSet: TrainingSample[];
  testSet: TrainingSample[];
} {
  const allSamples = loadRealMnistSamples();

  // Group by digit to ensure balanced stratification
  const byDigit: Record<number, MnistSampleInfo[]> = {};
  for (let d = 0; d < 10; d++) byDigit[d] = [];
  for (const s of allSamples) {
    byDigit[s.label].push(s);
  }

  const trainSet: TrainingSample[] = [];
  const testSet: TrainingSample[] = [];

  for (let d = 0; d < 10; d++) {
    const list = byDigit[d];
    const splitIndex = Math.floor(list.length * trainRatio);
    for (let i = 0; i < list.length; i++) {
      const item = list[i];
      const sample: TrainingSample = {
        id: item.id,
        label: item.label,
        pixels: item.pixels,
      };
      if (i < splitIndex) {
        trainSet.push(sample);
      } else {
        testSet.push(sample);
      }
    }
  }

  // Shuffle train set
  for (let i = trainSet.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1));
    [trainSet[i], trainSet[j]] = [trainSet[j], trainSet[i]];
  }

  return { trainSet, testSet };
}
