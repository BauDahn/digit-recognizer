import { TrainingSample } from '../types';
import { IMG_PIXELS } from './imageOps';

/**
 * Compact binary dataset format used for the files in public/datasets/*.bin.gz:
 *
 *   uint32 (little endian)  count
 *   uint8[count]            labels
 *   uint8[count * 784]      images, 28x28 row-major, 0 = background, 255 = ink
 *
 * The file is gzip-compressed; the browser decompresses it with DecompressionStream.
 */

export interface RawDataset {
  count: number;
  labels: Uint8Array;
  images: Uint8Array;
}

export function encodeDataset(labels: ArrayLike<number>, images: Uint8Array): Uint8Array {
  const count = labels.length;
  if (images.length !== count * IMG_PIXELS) {
    throw new Error(`Expected ${count * IMG_PIXELS} image bytes, got ${images.length}`);
  }
  const out = new Uint8Array(4 + count + images.length);
  new DataView(out.buffer).setUint32(0, count, true);
  out.set(Array.from(labels), 4);
  out.set(images, 4 + count);
  return out;
}

export function decodeDataset(bytes: Uint8Array): RawDataset {
  const count = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength).getUint32(0, true);
  const expected = 4 + count + count * IMG_PIXELS;
  if (bytes.length !== expected) {
    throw new Error(`Corrupt dataset: expected ${expected} bytes, got ${bytes.length}`);
  }
  return {
    count,
    labels: bytes.subarray(4, 4 + count),
    images: bytes.subarray(4 + count),
  };
}

export function imageAt(ds: RawDataset, index: number): Float32Array {
  const pixels = new Float32Array(IMG_PIXELS);
  const offset = index * IMG_PIXELS;
  for (let i = 0; i < IMG_PIXELS; i++) pixels[i] = ds.images[offset + i] / 255;
  return pixels;
}

export function toTrainingSamples(ds: RawDataset, idPrefix: string): TrainingSample[] {
  const samples: TrainingSample[] = [];
  for (let i = 0; i < ds.count; i++) {
    samples.push({ id: `${idPrefix}-${i}`, label: ds.labels[i], pixels: imageAt(ds, i) });
  }
  return samples;
}

/** Download and decompress a dataset from public/ (browser only). */
export async function fetchDataset(url: string): Promise<RawDataset> {
  const response = await fetch(url);
  if (!response.ok) {
    throw new Error(`Could not load ${url}: ${response.status} ${response.statusText}`);
  }
  let bytes = new Uint8Array(await response.arrayBuffer());
  // Some servers (e.g. Vite's dev server) send .gz files with "Content-Encoding: gzip",
  // so the browser has already decompressed them. Only decompress if the gzip magic bytes are there.
  if (bytes[0] === 0x1f && bytes[1] === 0x8b) {
    const stream = new Blob([bytes]).stream().pipeThrough(new DecompressionStream('gzip'));
    bytes = new Uint8Array(await new Response(stream).arrayBuffer());
  }
  return decodeDataset(bytes);
}

export type DatasetName = 'mnist-train' | 'mnist-test' | 'emnist-balanced-train' | 'emnist-balanced-test';

/** URL of a dataset file in public/datasets, respecting Vite's base path. */
export function datasetUrl(name: DatasetName): string {
  return `${import.meta.env.BASE_URL}datasets/${name}.bin.gz`;
}
