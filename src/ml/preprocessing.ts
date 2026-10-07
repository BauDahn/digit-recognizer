import { Box, centerInk28, IMG_PIXELS, inkFromRGBA } from './imageOps';

/**
 * Standard MNIST-style Preprocessing Pipeline for a drawing canvas:
 * 1. Read the canvas pixels as "ink" (dark on light background)
 * 2. Crop to the bounding box of the ink
 * 3. Scale into a 20x20 box preserving aspect ratio
 * 4. Center within a 28x28 matrix by Center of Mass
 *
 * Steps 2-4 live in the DOM-free `centerInk28` so they can be unit-tested and reused.
 */
export function preprocessCanvasTo28x28(
  sourceCanvas: HTMLCanvasElement,
  cropRect?: Box
): { pixels: Float32Array; boundingBox: Box | null } {
  const ctx = sourceCanvas.getContext('2d', { willReadFrequently: true });
  if (!ctx) {
    return { pixels: new Float32Array(IMG_PIXELS), boundingBox: null };
  }

  const sx = cropRect ? Math.max(0, Math.floor(cropRect.x)) : 0;
  const sy = cropRect ? Math.max(0, Math.floor(cropRect.y)) : 0;
  const sw = cropRect ? Math.min(sourceCanvas.width - sx, Math.ceil(cropRect.width)) : sourceCanvas.width;
  const sh = cropRect ? Math.min(sourceCanvas.height - sy, Math.ceil(cropRect.height)) : sourceCanvas.height;

  if (sw <= 0 || sh <= 0) {
    return { pixels: new Float32Array(IMG_PIXELS), boundingBox: null };
  }

  const ink = inkFromRGBA(ctx.getImageData(sx, sy, sw, sh).data, sw, sh);
  const { pixels, boundingBox } = centerInk28(ink, sw, sh);
  if (!boundingBox) return { pixels, boundingBox: null };

  return {
    pixels,
    boundingBox: { ...boundingBox, x: sx + boundingBox.x, y: sy + boundingBox.y },
  };
}
