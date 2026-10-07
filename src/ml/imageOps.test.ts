import { describe, expect, it } from 'vitest';
import { augment28, centerInk28, IMG_PIXELS, inkBoundingBox, rasterizeStrokes28, seededRandom } from './imageOps';

function centerOfMass(pixels: Float32Array) {
  let m = 0,
    sx = 0,
    sy = 0;
  for (let i = 0; i < IMG_PIXELS; i++) {
    m += pixels[i];
    sx += (i % 28) * pixels[i];
    sy += Math.floor(i / 28) * pixels[i];
  }
  return { x: sx / m, y: sy / m };
}

describe('centerInk28', () => {
  it('returns an empty image for a blank input', () => {
    const { pixels, boundingBox } = centerInk28(new Float32Array(100 * 100), 100, 100);
    expect(boundingBox).toBeNull();
    expect(pixels.every((v) => v === 0)).toBe(true);
  });

  it('fits the ink in a 20px box and centers it by center of mass', () => {
    // A filled rectangle in the top-left corner of a 200x100 image
    const w = 200,
      h = 100;
    const ink = new Float32Array(w * h);
    for (let y = 5; y < 45; y++) for (let x = 10; x < 30; x++) ink[y * w + x] = 1;

    const { pixels, boundingBox } = centerInk28(ink, w, h);
    expect(boundingBox).toEqual({ x: 10, y: 5, width: 20, height: 40 });

    const box = inkBoundingBox(pixels, 28, 28, 0.5)!;
    expect(box.height).toBeGreaterThanOrEqual(19);
    expect(box.height).toBeLessThanOrEqual(21);
    expect(box.width).toBeGreaterThanOrEqual(9);
    expect(box.width).toBeLessThanOrEqual(11);

    const com = centerOfMass(pixels);
    expect(Math.abs(com.x - 13.5)).toBeLessThan(1);
    expect(Math.abs(com.y - 13.5)).toBeLessThan(1);
  });
});

describe('rasterizeStrokes28', () => {
  it('draws the same symbol identically regardless of the size it was written at', () => {
    const plus = (s: number) => [
      [{ x: 0, y: s / 2 }, { x: s, y: s / 2 }],
      [{ x: s / 2, y: 0 }, { x: s / 2, y: s }],
    ];
    const small = rasterizeStrokes28(plus(30));
    const large = rasterizeStrokes28(plus(300));
    const diff = small.reduce((acc, v, i) => acc + Math.abs(v - large[i]), 0);
    expect(diff).toBeLessThan(1);
    expect(small.reduce((a, b) => a + b, 0)).toBeGreaterThan(20);
  });
});

describe('augment28', () => {
  it('keeps the image size and roughly the amount of ink', () => {
    const img = rasterizeStrokes28([[{ x: 0, y: 0 }, { x: 0, y: 10 }]]);
    const out = augment28(img, { thickenProb: 0 }, seededRandom(3));
    const sum = (a: Float32Array) => a.reduce((x, y) => x + y, 0);
    expect(out).toHaveLength(IMG_PIXELS);
    expect(sum(out) / sum(img)).toBeGreaterThan(0.7);
    expect(sum(out) / sum(img)).toBeLessThan(1.4);
  });
});
