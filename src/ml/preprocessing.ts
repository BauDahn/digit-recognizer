/**
 * Standard MNIST-style Preprocessing Pipeline:
 * 1. Find bounding box of drawn ink
 * 2. Crop to bounding box
 * 3. Scale into 20x20 box preserving aspect ratio
 * 4. Center within 28x28 matrix by Center of Mass
 */

export function preprocessCanvasTo28x28(
  sourceCanvas: HTMLCanvasElement,
  cropRect?: { x: number; y: number; width: number; height: number }
): { pixels: Float32Array; boundingBox: { x: number; y: number; width: number; height: number } | null } {
  const ctx = sourceCanvas.getContext('2d', { willReadFrequently: true });
  if (!ctx) {
    return { pixels: new Float32Array(784), boundingBox: null };
  }

  const sx = cropRect ? Math.max(0, Math.floor(cropRect.x)) : 0;
  const sy = cropRect ? Math.max(0, Math.floor(cropRect.y)) : 0;
  const sw = cropRect ? Math.min(sourceCanvas.width - sx, Math.ceil(cropRect.width)) : sourceCanvas.width;
  const sh = cropRect ? Math.min(sourceCanvas.height - sy, Math.ceil(cropRect.height)) : sourceCanvas.height;

  if (sw <= 0 || sh <= 0) {
    return { pixels: new Float32Array(784), boundingBox: null };
  }

  const imgData = ctx.getImageData(sx, sy, sw, sh);
  const data = imgData.data;

  // Find tight bounding box of ink pixels
  let minX = sw;
  let minY = sh;
  let maxX = -1;
  let maxY = -1;
  let totalInk = 0;

  // Check whether background is light (ink is dark) or dark (ink is light)
  // We compute ink intensity where 1.0 = full stroke, 0.0 = background
  const inkMatrix = new Float32Array(sw * sh);

  for (let y = 0; y < sh; y++) {
    for (let x = 0; x < sw; x++) {
      const idx = (y * sw + x) * 4;
      const r = data[idx];
      const g = data[idx + 1];
      const b = data[idx + 2];
      const a = data[idx + 3] / 255;

      // Ink detection: either alpha > 0 (transparent background) or darkness (white canvas)
      let intensity = 0;
      if (a > 0.05) {
        // If image has alpha
        const brightness = (0.299 * r + 0.587 * g + 0.114 * b) / 255;
        // Darkness is 1 - brightness
        intensity = (1 - brightness) * a;
      }

      inkMatrix[y * sw + x] = intensity;

      if (intensity > 0.12) {
        if (x < minX) minX = x;
        if (x > maxX) maxX = x;
        if (y < minY) minY = y;
        if (y > maxY) maxY = y;
        totalInk += intensity;
      }
    }
  }

  if (maxX < minX || maxY < minY || totalInk < 0.5) {
    // Empty canvas
    return { pixels: new Float32Array(784), boundingBox: null };
  }

  const bboxWidth = maxX - minX + 1;
  const bboxHeight = maxY - minY + 1;

  // Render cropped ink to temporary offscreen 20x20 area
  const cropCanvas = document.createElement('canvas');
  cropCanvas.width = bboxWidth;
  cropCanvas.height = bboxHeight;
  const cropCtx = cropCanvas.getContext('2d');
  if (!cropCtx) return { pixels: new Float32Array(784), boundingBox: null };

  const cropImgData = cropCtx.createImageData(bboxWidth, bboxHeight);
  for (let cy = 0; cy < bboxHeight; cy++) {
    for (let cx = 0; cx < bboxWidth; cx++) {
      const origX = minX + cx;
      const origY = minY + cy;
      const intensity = inkMatrix[origY * sw + origX];
      const cIdx = (cy * bboxWidth + cx) * 4;
      const byteVal = Math.min(255, Math.floor(intensity * 255));
      cropImgData.data[cIdx] = byteVal;
      cropImgData.data[cIdx + 1] = byteVal;
      cropImgData.data[cIdx + 2] = byteVal;
      cropImgData.data[cIdx + 3] = byteVal > 15 ? 255 : 0;
    }
  }
  cropCtx.putImageData(cropImgData, 0, 0);

  // Scale into 20x20 box preserving aspect ratio
  const scale = 20 / Math.max(bboxWidth, bboxHeight);
  const targetW = Math.max(1, Math.round(bboxWidth * scale));
  const targetH = Math.max(1, Math.round(bboxHeight * scale));

  const scaledCanvas = document.createElement('canvas');
  scaledCanvas.width = targetW;
  scaledCanvas.height = targetH;
  const scaledCtx = scaledCanvas.getContext('2d');
  if (!scaledCtx) return { pixels: new Float32Array(784), boundingBox: null };

  scaledCtx.imageSmoothingEnabled = true;
  scaledCtx.imageSmoothingQuality = 'high';
  scaledCtx.drawImage(cropCanvas, 0, 0, targetW, targetH);

  const scaledData = scaledCtx.getImageData(0, 0, targetW, targetH).data;

  // Compute Center of Mass within the 20x20 scaled image
  let sumX = 0;
  let sumY = 0;
  let totalMass = 0;
  const massGrid = new Float32Array(targetW * targetH);

  for (let y = 0; y < targetH; y++) {
    for (let x = 0; x < targetW; x++) {
      const val = scaledData[(y * targetW + x) * 4] / 255;
      massGrid[y * targetW + x] = val;
      sumX += x * val;
      sumY += y * val;
      totalMass += val;
    }
  }

  const comX = totalMass > 0 ? sumX / totalMass : targetW / 2;
  const comY = totalMass > 0 ? sumY / totalMass : targetH / 2;

  // Place into 28x28 matrix so Center of Mass aligns with center (13.5, 13.5)
  const destX = Math.round(13.5 - comX);
  const destY = Math.round(13.5 - comY);

  const finalPixels = new Float32Array(784); // 28x28

  for (let y = 0; y < targetH; y++) {
    for (let x = 0; x < targetW; x++) {
      const fx = destX + x;
      const fy = destY + y;
      if (fx >= 0 && fx < 28 && fy >= 0 && fy < 28) {
        finalPixels[fy * 28 + fx] = massGrid[y * targetW + x];
      }
    }
  }

  return {
    pixels: finalPixels,
    boundingBox: {
      x: sx + minX,
      y: sy + minY,
      width: bboxWidth,
      height: bboxHeight,
    },
  };
}
