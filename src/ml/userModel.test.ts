import { readFileSync } from 'node:fs';
import path from 'node:path';
import { gunzipSync } from 'node:zlib';
import { describe, expect, it, vi } from 'vitest';
import { decodeDataset, imageAt, toTrainingSamples } from './datasets';
import { seededRandom } from './imageOps';
import { fromModelFile, ModelFile } from './modelFile';
import { Correction, fineTune, toBytePixels } from './userModel';

const root = path.resolve(import.meta.dirname, '../..');
const load = (name: string) => decodeDataset(new Uint8Array(gunzipSync(readFileSync(path.join(root, `public/datasets/${name}.bin.gz`)))));

describe('fineTune (teaching the digits network from a correction)', () => {
  const model = fromModelFile(JSON.parse(readFileSync(path.join(root, 'public/models/digits.json'), 'utf8')) as ModelFile);
  const train = toTrainingSamples(load('mnist-train'), 'train');
  const test = load('mnist-test');
  const testSamples = toTrainingSamples(test, 'test');

  it('learns a corrected symbol without forgetting the others', () => {
    const net = fromModelFile(JSON.parse(readFileSync(path.join(root, 'public/models/digits.json'), 'utf8')) as ModelFile).network;
    const accuracyBefore = net.evaluate(testSamples).accuracy;

    // A "7" that the user insists is a "1": silly, but it is exactly what a correction looks like to the network
    const index = Array.from(test.labels).findIndex((l) => l === 7);
    const correction: Correction = { id: 'c1', char: '1', pixels: toBytePixels(imageAt(test, index)), createdAt: 0 };
    // the network shuffles its mini-batches with Math.random: make the run repeatable
    const spy = vi.spyOn(Math, 'random').mockImplementation(seededRandom(11));
    const result = fineTune(net, model.classes, [correction], correction, train, {}, seededRandom(5));
    spy.mockRestore();

    expect(result.before.char).toBe('7');
    expect(result.after.char).toBe('1');
    expect(result.learned).toBe(true);
    expect(net.evaluate(testSamples).accuracy).toBeGreaterThan(accuracyBefore - 0.06);
  });

  it('keeps earlier corrections when a new one arrives', () => {
    const net = fromModelFile(JSON.parse(readFileSync(path.join(root, 'public/models/digits.json'), 'utf8')) as ModelFile).network;
    const pick = (digit: number, nth: number) => Array.from(test.labels).map((l, i) => (l === digit ? i : -1)).filter((i) => i >= 0)[nth];
    const first: Correction = { id: 'a', char: '3', pixels: toBytePixels(imageAt(test, pick(8, 0))), createdAt: 0 };
    const second: Correction = { id: 'b', char: '4', pixels: toBytePixels(imageAt(test, pick(9, 0))), createdAt: 1 };
    fineTune(net, model.classes, [first], first, train, {}, seededRandom(1));
    fineTune(net, model.classes, [first, second], second, train, {}, seededRandom(2));
    const read = (c: Correction) => model.classes[net.predict(new Float32Array(Array.from(c.pixels, (v) => v / 255))).digit];
    expect(read(first)).toBe('3');
    expect(read(second)).toBe('4');
  });

  it('refuses classes the network does not have', () => {
    const net = fromModelFile(JSON.parse(readFileSync(path.join(root, 'public/models/digits.json'), 'utf8')) as ModelFile).network;
    const c: Correction = { id: 'x', char: 'Σ', pixels: new Uint8Array(784), createdAt: 0 };
    expect(() => fineTune(net, model.classes, [c], c, train)).toThrow(/not one of the symbols/);
  });
});
