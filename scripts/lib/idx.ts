/**
 * Node helpers to download MNIST / EMNIST and read their IDX files.
 * Raw downloads are cached in .cache/datasets (gitignored).
 */
import { execFileSync } from 'node:child_process';
import { existsSync, mkdirSync, readFileSync, writeFileSync } from 'node:fs';
import path from 'node:path';
import { gunzipSync } from 'node:zlib';
import { IMG_PIXELS } from '../../src/ml/imageOps';

export const CACHE_DIR = path.resolve(import.meta.dirname, '../../.cache/datasets');

const MNIST_BASE = 'https://ossci-datasets.s3.amazonaws.com/mnist';
const EMNIST_ZIP_URL = 'https://biometrics.nist.gov/cs_links/EMNIST/gzip.zip';

export interface IdxDataset {
  labels: Uint8Array;
  images: Uint8Array; // count * 784, 0..255
}

async function downloadCached(url: string, fileName: string): Promise<string> {
  mkdirSync(CACHE_DIR, { recursive: true });
  const target = path.join(CACHE_DIR, fileName);
  if (existsSync(target)) return target;
  console.log(`Downloading ${url} ...`);
  const response = await fetch(url);
  if (!response.ok) throw new Error(`Download failed: ${url} (${response.status})`);
  writeFileSync(target, new Uint8Array(await response.arrayBuffer()));
  return target;
}

function parseIdx(buffer: Uint8Array, expectedMagic: number): { dims: number[]; data: Uint8Array } {
  const view = new DataView(buffer.buffer, buffer.byteOffset, buffer.byteLength);
  const magic = view.getUint32(0, false);
  if (magic !== expectedMagic) throw new Error(`Bad IDX magic ${magic}, expected ${expectedMagic}`);
  const nDims = magic & 0xff;
  const dims: number[] = [];
  for (let i = 0; i < nDims; i++) dims.push(view.getUint32(4 + i * 4, false));
  return { dims, data: buffer.subarray(4 + nDims * 4) };
}

function readIdxPair(imagesGz: Uint8Array, labelsGz: Uint8Array): IdxDataset {
  const images = parseIdx(gunzipSync(imagesGz), 0x00000803);
  const labels = parseIdx(gunzipSync(labelsGz), 0x00000801);
  if (images.dims[0] !== labels.dims[0] || images.dims[1] * images.dims[2] !== IMG_PIXELS) {
    throw new Error('Unexpected IDX dimensions');
  }
  return { labels: labels.data, images: images.data };
}

export async function loadMnist(split: 'train' | 'test'): Promise<IdxDataset> {
  const prefix = split === 'train' ? 'train' : 't10k';
  const imgs = await downloadCached(`${MNIST_BASE}/${prefix}-images-idx3-ubyte.gz`, `mnist-${prefix}-images.gz`);
  const lbls = await downloadCached(`${MNIST_BASE}/${prefix}-labels-idx1-ubyte.gz`, `mnist-${prefix}-labels.gz`);
  return readIdxPair(readFileSync(imgs), readFileSync(lbls));
}

/** EMNIST Balanced. Requires the `unzip` command (the official archive is a ~560MB zip). */
export async function loadEmnistBalanced(split: 'train' | 'test'): Promise<IdxDataset> {
  const zip = await downloadCached(EMNIST_ZIP_URL, 'emnist-gzip.zip');
  const extract = (name: string) =>
    new Uint8Array(execFileSync('unzip', ['-p', zip, `gzip/${name}`], { maxBuffer: 1 << 30 }));
  const ds = readIdxPair(
    extract(`emnist-balanced-${split}-images-idx3-ubyte.gz`),
    extract(`emnist-balanced-${split}-labels-idx1-ubyte.gz`)
  );
  // EMNIST images are stored transposed (column-major); flip them to match MNIST.
  const fixed = new Uint8Array(ds.images.length);
  for (let n = 0; n < ds.labels.length; n++) {
    const o = n * IMG_PIXELS;
    for (let y = 0; y < 28; y++) {
      for (let x = 0; x < 28; x++) fixed[o + y * 28 + x] = ds.images[o + x * 28 + y];
    }
  }
  return { labels: ds.labels, images: fixed };
}

/** Take up to `perClass` samples of every class, preserving dataset order (deterministic). */
export function stratifiedSubset(ds: IdxDataset, perClass: number, numClasses: number): IdxDataset {
  const counts = new Array(numClasses).fill(0);
  const picked: number[] = [];
  for (let i = 0; i < ds.labels.length; i++) {
    const label = ds.labels[i];
    if (counts[label] < perClass) {
      counts[label]++;
      picked.push(i);
    }
  }
  const labels = new Uint8Array(picked.length);
  const images = new Uint8Array(picked.length * IMG_PIXELS);
  picked.forEach((src, dst) => {
    labels[dst] = ds.labels[src];
    images.set(ds.images.subarray(src * IMG_PIXELS, (src + 1) * IMG_PIXELS), dst * IMG_PIXELS);
  });
  return { labels, images };
}
