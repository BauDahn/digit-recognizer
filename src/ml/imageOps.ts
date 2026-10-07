/**
 * Pure (DOM-free) image operations shared by the browser UI, the training worker,
 * the dataset scripts and the tests.
 *
 * All "ink" images are Float32Arrays in row-major order where 1 = ink and 0 = background,
 * which is the same convention MNIST uses (white digit on black background).
 */

export const IMG_SIZE = 28;
export const IMG_PIXELS = IMG_SIZE * IMG_SIZE;
/** MNIST fits each digit in a 20x20 box and then centers it by center of mass in 28x28. */
const INNER_BOX = 20;

export interface Box {
  x: number;
  y: number;
  width: number;
  height: number;
}

export interface StrokePoint {
  x: number;
  y: number;
}

/**
 * Convert RGBA pixel data (e.g. from a canvas) into an ink matrix.
 * Dark pixels on a light/transparent background count as ink.
 */
export function inkFromRGBA(data: ArrayLike<number>, width: number, height: number): Float32Array {
  const ink = new Float32Array(width * height);
  for (let i = 0; i < width * height; i++) {
    const a = data[i * 4 + 3] / 255;
    if (a <= 0.05) continue;
    const brightness = (0.299 * data[i * 4] + 0.587 * data[i * 4 + 1] + 0.114 * data[i * 4 + 2]) / 255;
    ink[i] = (1 - brightness) * a;
  }
  return ink;
}

/** Tight bounding box of pixels above threshold, or null if the image is empty. */
export function inkBoundingBox(ink: Float32Array, width: number, height: number, threshold = 0.12): Box | null {
  let minX = width;
  let minY = height;
  let maxX = -1;
  let maxY = -1;
  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      if (ink[y * width + x] > threshold) {
        if (x < minX) minX = x;
        if (x > maxX) maxX = x;
        if (y < minY) minY = y;
        if (y > maxY) maxY = y;
      }
    }
  }
  if (maxX < 0) return null;
  return { x: minX, y: minY, width: maxX - minX + 1, height: maxY - minY + 1 };
}

/** Bilinear sample with zero padding outside the image. */
function sampleBilinear(img: Float32Array, width: number, height: number, x: number, y: number): number {
  const x0 = Math.floor(x);
  const y0 = Math.floor(y);
  const fx = x - x0;
  const fy = y - y0;
  const at = (xx: number, yy: number) =>
    xx >= 0 && xx < width && yy >= 0 && yy < height ? img[yy * width + xx] : 0;
  return (
    at(x0, y0) * (1 - fx) * (1 - fy) +
    at(x0 + 1, y0) * fx * (1 - fy) +
    at(x0, y0 + 1) * (1 - fx) * fy +
    at(x0 + 1, y0 + 1) * fx * fy
  );
}

/**
 * MNIST-style normalization:
 * 1. crop to the ink bounding box
 * 2. resize to fit a 20x20 box (keeping aspect ratio), averaging to avoid aliasing
 * 3. place in a 28x28 image so the center of mass sits at the center
 */
export function centerInk28(
  ink: Float32Array,
  width: number,
  height: number,
  threshold = 0.12
): { pixels: Float32Array; boundingBox: Box | null } {
  const pixels = new Float32Array(IMG_PIXELS);
  const box = inkBoundingBox(ink, width, height, threshold);
  if (!box) return { pixels, boundingBox: null };

  const scale = INNER_BOX / Math.max(box.width, box.height);
  const tw = Math.max(1, Math.round(box.width * scale));
  const th = Math.max(1, Math.round(box.height * scale));

  // Supersampled resize: each target pixel averages SS x SS bilinear samples of the source.
  const SS = Math.max(1, Math.min(6, Math.ceil(1 / scale)));
  const small = new Float32Array(tw * th);
  let mass = 0;
  let sumX = 0;
  let sumY = 0;
  for (let y = 0; y < th; y++) {
    for (let x = 0; x < tw; x++) {
      let acc = 0;
      for (let sy = 0; sy < SS; sy++) {
        for (let sx = 0; sx < SS; sx++) {
          const srcX = box.x + ((x + (sx + 0.5) / SS) / tw) * box.width - 0.5;
          const srcY = box.y + ((y + (sy + 0.5) / SS) / th) * box.height - 0.5;
          acc += sampleBilinear(ink, width, height, srcX, srcY);
        }
      }
      const v = Math.min(1, acc / (SS * SS));
      small[y * tw + x] = v;
      mass += v;
      sumX += x * v;
      sumY += y * v;
    }
  }

  const comX = mass > 0 ? sumX / mass : tw / 2;
  const comY = mass > 0 ? sumY / mass : th / 2;
  const destX = Math.round(13.5 - comX);
  const destY = Math.round(13.5 - comY);

  for (let y = 0; y < th; y++) {
    for (let x = 0; x < tw; x++) {
      const fx = destX + x;
      const fy = destY + y;
      if (fx >= 0 && fx < IMG_SIZE && fy >= 0 && fy < IMG_SIZE) {
        pixels[fy * IMG_SIZE + fx] = small[y * tw + x];
      }
    }
  }

  return { pixels, boundingBox: box };
}

/**
 * Rasterize vector strokes into a 28x28 MNIST-style image.
 * The geometry is scaled first and drawn with a fixed pen thickness afterwards, so
 * a symbol written small or large on the canvas ends up with MNIST-like stroke width.
 */
export function rasterizeStrokes28(strokes: StrokePoint[][], penRadius = 4.2): Float32Array {
  const pts = strokes.flat();
  if (pts.length === 0) return new Float32Array(IMG_PIXELS);

  let minX = Infinity;
  let minY = Infinity;
  let maxX = -Infinity;
  let maxY = -Infinity;
  for (const p of pts) {
    if (p.x < minX) minX = p.x;
    if (p.x > maxX) maxX = p.x;
    if (p.y < minY) minY = p.y;
    if (p.y > maxY) maxY = p.y;
  }

  // Draw at 4x resolution (80px inner box) and let centerInk28 downsample.
  const inner = INNER_BOX * 4;
  const extent = Math.max(maxX - minX, maxY - minY, 1e-6);
  const scale = inner / extent;
  const pad = Math.ceil(penRadius) + 1;
  const size = inner + pad * 2;
  const grid = new Float32Array(size * size);
  // Center the shorter dimension inside the square drawing area.
  const offX = pad + (inner - (maxX - minX) * scale) / 2;
  const offY = pad + (inner - (maxY - minY) * scale) / 2;

  const toGrid = (p: StrokePoint) => ({ x: (p.x - minX) * scale + offX, y: (p.y - minY) * scale + offY });

  for (const stroke of strokes) {
    if (stroke.length === 0) continue;
    const gp = stroke.map(toGrid);
    if (gp.length === 1) gp.push(gp[0]);
    for (let i = 1; i < gp.length; i++) {
      drawSegment(grid, size, gp[i - 1], gp[i], penRadius);
    }
  }

  return centerInk28(grid, size, size, 0.05).pixels;
}

function drawSegment(grid: Float32Array, size: number, a: StrokePoint, b: StrokePoint, r: number) {
  const x0 = Math.max(0, Math.floor(Math.min(a.x, b.x) - r - 1));
  const x1 = Math.min(size - 1, Math.ceil(Math.max(a.x, b.x) + r + 1));
  const y0 = Math.max(0, Math.floor(Math.min(a.y, b.y) - r - 1));
  const y1 = Math.min(size - 1, Math.ceil(Math.max(a.y, b.y) + r + 1));
  const dx = b.x - a.x;
  const dy = b.y - a.y;
  const len2 = dx * dx + dy * dy;
  for (let y = y0; y <= y1; y++) {
    for (let x = x0; x <= x1; x++) {
      const px = x + 0.5;
      const py = y + 0.5;
      let t = len2 > 0 ? ((px - a.x) * dx + (py - a.y) * dy) / len2 : 0;
      t = Math.max(0, Math.min(1, t));
      const ex = a.x + t * dx - px;
      const ey = a.y + t * dy - py;
      const d = Math.sqrt(ex * ex + ey * ey);
      // Soft edge of 1px for anti-aliasing
      const v = Math.max(0, Math.min(1, r - d + 0.5));
      const idx = y * size + x;
      if (v > grid[idx]) grid[idx] = v;
    }
  }
}

export interface AugmentOptions {
  maxShift?: number; // pixels
  maxRotateDeg?: number;
  scaleRange?: number; // e.g. 0.1 => scale in [0.9, 1.1]
  thickenProb?: number; // probability of a 3x3 dilation
}

export const DEFAULT_AUGMENT: Required<AugmentOptions> = {
  maxShift: 2,
  maxRotateDeg: 10,
  scaleRange: 0.1,
  thickenProb: 0.3,
};

/**
 * Data augmentation: a random small affine transform (rotation, scale, shift) and
 * optionally a thicker stroke. Produces "new" training images from existing ones.
 */
export function augment28(
  pixels: Float32Array,
  options: AugmentOptions = {},
  random: () => number = Math.random
): Float32Array {
  const o = { ...DEFAULT_AUGMENT, ...options };
  const angle = ((random() * 2 - 1) * o.maxRotateDeg * Math.PI) / 180;
  const scale = 1 + (random() * 2 - 1) * o.scaleRange;
  const shiftX = (random() * 2 - 1) * o.maxShift;
  const shiftY = (random() * 2 - 1) * o.maxShift;
  const cos = Math.cos(angle) / scale;
  const sin = Math.sin(angle) / scale;
  const c = 13.5;

  let out: Float32Array = new Float32Array(IMG_PIXELS);
  for (let y = 0; y < IMG_SIZE; y++) {
    for (let x = 0; x < IMG_SIZE; x++) {
      // Inverse mapping: for each output pixel find where it comes from in the input.
      const dx = x - c - shiftX;
      const dy = y - c - shiftY;
      const srcX = cos * dx + sin * dy + c;
      const srcY = -sin * dx + cos * dy + c;
      out[y * IMG_SIZE + x] = sampleBilinear(pixels, IMG_SIZE, IMG_SIZE, srcX, srcY);
    }
  }

  if (random() < o.thickenProb) out = dilate28(out);
  return out;
}

/** 3x3 max filter blended with the original: makes strokes slightly thicker. */
export function dilate28(pixels: Float32Array): Float32Array {
  const out = new Float32Array(IMG_PIXELS);
  for (let y = 0; y < IMG_SIZE; y++) {
    for (let x = 0; x < IMG_SIZE; x++) {
      let m = 0;
      for (let dy = -1; dy <= 1; dy++) {
        for (let dx = -1; dx <= 1; dx++) {
          const xx = x + dx;
          const yy = y + dy;
          if (xx >= 0 && xx < IMG_SIZE && yy >= 0 && yy < IMG_SIZE) {
            const v = pixels[yy * IMG_SIZE + xx] * (dx === 0 || dy === 0 ? 1 : 0.7);
            if (v > m) m = v;
          }
        }
      }
      out[y * IMG_SIZE + x] = Math.max(pixels[y * IMG_SIZE + x], m * 0.85);
    }
  }
  return out;
}

/** Deterministic PRNG (mulberry32) so scripts and tests are reproducible. */
export function seededRandom(seed: number): () => number {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}
