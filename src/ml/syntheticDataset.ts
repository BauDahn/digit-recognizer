import { TrainingSample } from '../types';

/**
 * Generates synthetic and stroke-based training samples for digits 0-9
 * with morphological variations, random slants, scales, and offsets.
 */

// Basic stroke skeleton curves for digits 0-9 normalized in [0, 1] range
interface PathSegment {
  type: 'line' | 'quad' | 'cubic' | 'arc';
  pts: number[];
}

interface DigitSkeleton {
  paths: PathSegment[];
}

const DIGIT_SKELETONS: Record<number, DigitSkeleton[]> = {
  0: [
    // Standard oval 0
    {
      paths: [
        { type: 'arc', pts: [0.5, 0.5, 0.32, 0.42, 0, Math.PI * 2] },
      ],
    },
    // Elliptical slender 0
    {
      paths: [
        { type: 'arc', pts: [0.5, 0.5, 0.24, 0.44, 0, Math.PI * 2] },
      ],
    },
  ],
  1: [
    // Straight vertical 1
    {
      paths: [
        { type: 'line', pts: [0.5, 0.1, 0.5, 0.9] },
      ],
    },
    // European 1 with top hook and base
    {
      paths: [
        { type: 'line', pts: [0.35, 0.3, 0.5, 0.1] },
        { type: 'line', pts: [0.5, 0.1, 0.5, 0.9] },
        { type: 'line', pts: [0.3, 0.9, 0.7, 0.9] },
      ],
    },
  ],
  2: [
    // Classical 2
    {
      paths: [
        { type: 'cubic', pts: [0.25, 0.3, 0.25, 0.1, 0.75, 0.1, 0.75, 0.35] },
        { type: 'line', pts: [0.75, 0.35, 0.25, 0.9] },
        { type: 'line', pts: [0.25, 0.9, 0.75, 0.9] },
      ],
    },
    // Curvy 2 with loop
    {
      paths: [
        { type: 'cubic', pts: [0.25, 0.25, 0.3, 0.08, 0.75, 0.1, 0.7, 0.35] },
        { type: 'quad', pts: [0.7, 0.35, 0.4, 0.65, 0.25, 0.88] },
        { type: 'cubic', pts: [0.25, 0.88, 0.4, 0.95, 0.6, 0.85, 0.8, 0.9] },
      ],
    },
  ],
  3: [
    // Round double-bump 3
    {
      paths: [
        { type: 'cubic', pts: [0.28, 0.2, 0.4, 0.08, 0.75, 0.12, 0.65, 0.45] },
        { type: 'cubic', pts: [0.65, 0.45, 0.8, 0.7, 0.4, 0.92, 0.25, 0.82] },
      ],
    },
    // Flat-top 3
    {
      paths: [
        { type: 'line', pts: [0.25, 0.15, 0.75, 0.15] },
        { type: 'line', pts: [0.75, 0.15, 0.48, 0.45] },
        { type: 'cubic', pts: [0.48, 0.45, 0.8, 0.55, 0.75, 0.88, 0.25, 0.85] },
      ],
    },
  ],
  4: [
    // Open 4
    {
      paths: [
        { type: 'line', pts: [0.65, 0.1, 0.2, 0.65] },
        { type: 'line', pts: [0.2, 0.65, 0.82, 0.65] },
        { type: 'line', pts: [0.65, 0.35, 0.65, 0.9] },
      ],
    },
    // Closed top 4
    {
      paths: [
        { type: 'line', pts: [0.55, 0.1, 0.25, 0.65] },
        { type: 'line', pts: [0.25, 0.65, 0.75, 0.65] },
        { type: 'line', pts: [0.55, 0.1, 0.55, 0.9] },
      ],
    },
  ],
  5: [
    // Classic 5
    {
      paths: [
        { type: 'line', pts: [0.72, 0.15, 0.3, 0.15] },
        { type: 'line', pts: [0.3, 0.15, 0.28, 0.45] },
        { type: 'cubic', pts: [0.28, 0.45, 0.78, 0.38, 0.78, 0.88, 0.25, 0.85] },
      ],
    },
  ],
  6: [
    // Smooth 6 with enclosed bottom loop
    {
      paths: [
        { type: 'cubic', pts: [0.65, 0.15, 0.35, 0.2, 0.25, 0.6, 0.25, 0.7] },
        { type: 'arc', pts: [0.5, 0.68, 0.26, 0.22, 0, Math.PI * 2] },
      ],
    },
  ],
  7: [
    // Standard 7
    {
      paths: [
        { type: 'line', pts: [0.25, 0.15, 0.75, 0.15] },
        { type: 'line', pts: [0.75, 0.15, 0.35, 0.9] },
      ],
    },
    // European crossed 7
    {
      paths: [
        { type: 'line', pts: [0.25, 0.15, 0.75, 0.15] },
        { type: 'line', pts: [0.75, 0.15, 0.38, 0.9] },
        { type: 'line', pts: [0.4, 0.52, 0.68, 0.52] },
      ],
    },
  ],
  8: [
    // Double loop 8
    {
      paths: [
        { type: 'arc', pts: [0.5, 0.32, 0.22, 0.22, 0, Math.PI * 2] },
        { type: 'arc', pts: [0.5, 0.68, 0.26, 0.26, 0, Math.PI * 2] },
      ],
    },
    // Figure-8 stroke
    {
      paths: [
        { type: 'cubic', pts: [0.5, 0.1, 0.2, 0.15, 0.78, 0.5, 0.5, 0.9] },
        { type: 'cubic', pts: [0.5, 0.9, 0.22, 0.5, 0.8, 0.15, 0.5, 0.1] },
      ],
    },
  ],
  9: [
    // Top-loop 9 with tail
    {
      paths: [
        { type: 'arc', pts: [0.5, 0.32, 0.25, 0.22, 0, Math.PI * 2] },
        { type: 'cubic', pts: [0.75, 0.32, 0.75, 0.65, 0.65, 0.85, 0.4, 0.92] },
      ],
    },
    // Straight-stem 9
    {
      paths: [
        { type: 'arc', pts: [0.5, 0.32, 0.24, 0.22, 0, Math.PI * 2] },
        { type: 'line', pts: [0.74, 0.32, 0.65, 0.9] },
      ],
    },
  ],
};

/**
 * Render a single synthetic digit onto a 28x28 Float32Array with random augmentations
 */
export function generateSyntheticDigit(
  digit: number,
  options?: {
    slant?: number; // radians (-0.25 to 0.25)
    lineWidth?: number; // 2 to 4.5
    scale?: number; // 0.8 to 1.15
    offsetX?: number; // -2 to 2
    offsetY?: number; // -2 to 2
    noise?: number; // 0 to 0.05
  }
): Float32Array {
  const canvas = document.createElement('canvas');
  canvas.width = 28;
  canvas.height = 28;
  const ctx = canvas.getContext('2d');
  if (!ctx) return new Float32Array(784);

  ctx.fillStyle = '#000000';
  ctx.fillRect(0, 0, 28, 28);

  const skeletons = DIGIT_SKELETONS[digit] || DIGIT_SKELETONS[0];
  const skeleton = skeletons[Math.floor(Math.random() * skeletons.length)];

  const slant = options?.slant ?? (Math.random() - 0.5) * 0.35;
  const scale = options?.scale ?? (0.85 + Math.random() * 0.25);
  const lineWidth = options?.lineWidth ?? (2.4 + Math.random() * 1.6);
  const offsetX = options?.offsetX ?? (Math.random() - 0.5) * 3;
  const offsetY = options?.offsetY ?? (Math.random() - 0.5) * 3;

  ctx.save();
  ctx.translate(14 + offsetX, 14 + offsetY);
  ctx.transform(1, 0, slant, 1, 0, 0); // shear/slant
  ctx.scale(scale, scale);
  ctx.translate(-14, -14);

  ctx.strokeStyle = '#ffffff';
  ctx.lineWidth = lineWidth;
  ctx.lineCap = 'round';
  ctx.lineJoin = 'round';

  for (const path of skeleton.paths) {
    ctx.beginPath();
    if (path.type === 'line') {
      const [x1, y1, x2, y2] = path.pts;
      ctx.moveTo(x1 * 28, y1 * 28);
      ctx.lineTo(x2 * 28, y2 * 28);
    } else if (path.type === 'quad') {
      const [x1, y1, cx, cy, x2, y2] = path.pts;
      ctx.moveTo(x1 * 28, y1 * 28);
      ctx.quadraticCurveTo(cx * 28, cy * 28, x2 * 28, y2 * 28);
    } else if (path.type === 'cubic') {
      const [x1, y1, c1x, c1y, c2x, c2y, x2, y2] = path.pts;
      ctx.moveTo(x1 * 28, y1 * 28);
      ctx.bezierCurveTo(c1x * 28, c1y * 28, c2x * 28, c2y * 28, x2 * 28, y2 * 28);
    } else if (path.type === 'arc') {
      const [cx, cy, rx, ry, sa, ea] = path.pts;
      ctx.ellipse(cx * 28, cy * 28, rx * 28, ry * 28, 0, sa, ea);
    }
    ctx.stroke();
  }

  ctx.restore();

  const imgData = ctx.getImageData(0, 0, 28, 28).data;
  const pixels = new Float32Array(784);
  const noise = options?.noise ?? 0.02;

  for (let i = 0; i < 784; i++) {
    let p = imgData[i * 4] / 255;
    if (noise > 0 && Math.random() < 0.15) {
      p = Math.max(0, Math.min(1, p + (Math.random() - 0.5) * noise));
    }
    pixels[i] = p;
  }

  return pixels;
}

/**
 * Generates an initial training dataset across digits 0-9
 */
export function createSyntheticDataset(samplesPerDigit = 35): TrainingSample[] {
  const dataset: TrainingSample[] = [];

  for (let digit = 0; digit <= 9; digit++) {
    for (let s = 0; s < samplesPerDigit; s++) {
      const pixels = generateSyntheticDigit(digit);
      dataset.push({
        id: `synth-${digit}-${s}`,
        label: digit,
        pixels,
      });
    }
  }

  // Shuffle dataset with Fisher-Yates
  for (let i = dataset.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1));
    [dataset[i], dataset[j]] = [dataset[j], dataset[i]];
  }

  return dataset;
}
