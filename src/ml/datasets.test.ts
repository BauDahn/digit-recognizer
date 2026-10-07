import { describe, expect, it } from 'vitest';
import { decodeDataset, encodeDataset, imageAt, toTrainingSamples } from './datasets';

describe('binary dataset format', () => {
  it('round-trips labels and images', () => {
    const labels = [3, 1, 4];
    const images = new Uint8Array(3 * 784).map((_, i) => i % 256);
    const ds = decodeDataset(encodeDataset(labels, images));
    expect(ds.count).toBe(3);
    expect(Array.from(ds.labels)).toEqual(labels);
    expect(imageAt(ds, 1)[0]).toBeCloseTo((784 % 256) / 255);
    const samples = toTrainingSamples(ds, 't');
    expect(samples.map((s) => s.label)).toEqual(labels);
  });

  it('detects truncated files', () => {
    const bytes = encodeDataset([1], new Uint8Array(784));
    expect(() => decodeDataset(bytes.subarray(0, bytes.length - 1))).toThrow(/Corrupt/);
  });
});
