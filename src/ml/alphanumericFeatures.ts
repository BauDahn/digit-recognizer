import { AlphanumericFeatureBreakdown, SegmentedCharacter } from '../types';

/**
 * Extracts invariant topological, geometric, and projection features from 28x28 normalized binary matrix
 */
export function extractAlphanumericFeatures(pixels: Float32Array): AlphanumericFeatureBreakdown {
  // Convert to binary grid (threshold at 0.25)
  const grid: boolean[][] = Array.from({ length: 28 }, (_, y) =>
    Array.from({ length: 28 }, (_, x) => pixels[y * 28 + x] > 0.2)
  );

  // 1. Loop / Hole Count via Flood-Fill on inverted background
  // Padded 30x30 grid to ensure exterior background connects around borders
  const paddedW = 30;
  const paddedH = 30;
  const visited: boolean[][] = Array.from({ length: paddedH }, () => Array(paddedW).fill(false));

  const isInk = (px: number, py: number) => {
    if (px < 1 || px > 28 || py < 1 || py > 28) return false;
    return grid[py - 1][px - 1];
  };

  // BFS Flood fill from outer border (0, 0)
  const queue: [number, number][] = [[0, 0]];
  visited[0][0] = true;

  while (queue.length > 0) {
    const [cx, cy] = queue.shift()!;
    const neighbors: [number, number][] = [
      [cx + 1, cy],
      [cx - 1, cy],
      [cx, cy + 1],
      [cx, cy - 1],
    ];

    for (const [nx, ny] of neighbors) {
      if (nx >= 0 && nx < paddedW && ny >= 0 && ny < paddedH) {
        if (!visited[ny][nx] && !isInk(nx, ny)) {
          visited[ny][nx] = true;
          queue.push([nx, ny]);
        }
      }
    }
  }

  // Count enclosed background components (holes)
  let loopCount = 0;
  for (let y = 1; y <= 28; y++) {
    for (let x = 1; x <= 28; x++) {
      if (!isInk(x, y) && !visited[y][x]) {
        // Found a hole component!
        let holeSize = 0;
        const holeQueue: [number, number][] = [[x, y]];
        visited[y][x] = true;

        while (holeQueue.length > 0) {
          const [hx, hy] = holeQueue.shift()!;
          holeSize++;
          const hNeighbors: [number, number][] = [
            [hx + 1, hy],
            [hx - 1, hy],
            [hx, hy + 1],
            [hx, hy - 1],
          ];
          for (const [nhx, nhy] of hNeighbors) {
            if (nhx >= 1 && nhx <= 28 && nhy >= 1 && nhy <= 28) {
              if (!visited[nhy][nhx] && !isInk(nhx, nhy)) {
                visited[nhy][nhx] = true;
                holeQueue.push([nhx, nhy]);
              }
            }
          }
        }

        // Filter out tiny 1-2 pixel noise holes
        if (holeSize >= 3) {
          loopCount++;
        }
      }
    }
  }

  // 2. Quadrant Densities & Mass Distribution
  let totalMass = 0;
  let qTL = 0,
    qTR = 0,
    qBL = 0,
    qBR = 0;
  let massLeft = 0,
    massRight = 0,
    massTop = 0,
    massBottom = 0;

  for (let y = 0; y < 28; y++) {
    for (let x = 0; x < 28; x++) {
      const val = pixels[y * 28 + x];
      if (val > 0.05) {
        totalMass += val;
        if (x < 14 && y < 14) qTL += val;
        else if (x >= 14 && y < 14) qTR += val;
        else if (x < 14 && y >= 14) qBL += val;
        else qBR += val;

        if (x < 14) massLeft += val;
        else massRight += val;

        if (y < 14) massTop += val;
        else massBottom += val;
      }
    }
  }

  const safeMass = Math.max(0.1, totalMass);
  const densityTop = massTop / safeMass;
  const densityBottom = massBottom / safeMass;
  const densityLeft = massLeft / safeMass;
  const densityRight = massRight / safeMass;

  // 3. Symmetry Indexes
  let hSymDiff = 0;
  let vSymDiff = 0;
  for (let y = 0; y < 28; y++) {
    for (let x = 0; x < 14; x++) {
      hSymDiff += Math.abs(pixels[y * 28 + x] - pixels[y * 28 + (27 - x)]);
    }
  }
  for (let y = 0; y < 14; y++) {
    for (let x = 0; x < 28; x++) {
      vSymDiff += Math.abs(pixels[y * 28 + x] - pixels[(27 - y) * 28 + x]);
    }
  }

  const horizontalSymmetry = Math.max(0, 1 - (hSymDiff * 2) / safeMass);
  const verticalSymmetry = Math.max(0, 1 - (vSymDiff * 2) / safeMass);

  // 4. Endpoints & Junctions estimation via transition crossings
  let endpointEstimate = 0;
  let crossEstimate = 0;

  for (let y = 2; y < 26; y++) {
    for (let x = 2; x < 26; x++) {
      if (grid[y][x]) {
        // Count 8-neighborhood transitions
        const n = [
          grid[y - 1][x],
          grid[y - 1][x + 1],
          grid[y][x + 1],
          grid[y + 1][x + 1],
          grid[y + 1][x],
          grid[y + 1][x - 1],
          grid[y][x - 1],
          grid[y - 1][x - 1],
        ];

        let neighborsCount = 0;
        let transitions = 0;
        for (let i = 0; i < 8; i++) {
          if (n[i]) neighborsCount++;
          if (!n[i] && n[(i + 1) % 8]) transitions++;
        }

        if (neighborsCount === 1) endpointEstimate++;
        if (transitions >= 3) crossEstimate++;
      }
    }
  }

  // Find aspect ratio of ink
  let minX = 28,
    maxX = 0,
    minY = 28,
    maxY = 0;
  for (let y = 0; y < 28; y++) {
    for (let x = 0; x < 28; x++) {
      if (grid[y][x]) {
        if (x < minX) minX = x;
        if (x > maxX) maxX = x;
        if (y < minY) minY = y;
        if (y > maxY) maxY = y;
      }
    }
  }
  const spanW = Math.max(1, maxX - minX + 1);
  const spanH = Math.max(1, maxY - minY + 1);
  const aspectRatio = spanW / spanH;

  return {
    loopCount,
    endpointCount: Math.min(6, Math.round(endpointEstimate / 2.5)),
    crossIntersectionCount: Math.min(4, Math.round(crossEstimate / 3)),
    aspectRatio,
    horizontalSymmetry,
    verticalSymmetry,
    densityTop,
    densityBottom,
    densityLeft,
    densityRight,
  };
}

/**
 * Knowledge Base Character Prototype Specs
 */
interface CharPrototype {
  char: string;
  category: 'digit' | 'uppercase' | 'lowercase';
  expectedLoops: number[]; // e.g. [1] or [0, 1]
  expectedAspect: number; // typical w/h
  typicalCrosses?: number;
  topHeavy?: boolean;
  bottomHeavy?: boolean;
  notes: string;
}

const CHAR_PROTOTYPES: CharPrototype[] = [
  // Digits
  { char: '0', category: 'digit', expectedLoops: [1], expectedAspect: 0.7, notes: 'Single central loop, highly symmetric' },
  { char: '1', category: 'digit', expectedLoops: [0], expectedAspect: 0.35, notes: 'Slender vertical stroke, no loops' },
  { char: '2', category: 'digit', expectedLoops: [0], expectedAspect: 0.65, bottomHeavy: true, notes: 'Top hook + horizontal base' },
  { char: '3', category: 'digit', expectedLoops: [0], expectedAspect: 0.6, notes: 'Open right lobes, no loops' },
  { char: '4', category: 'digit', expectedLoops: [0, 1], expectedAspect: 0.75, typicalCrosses: 1, notes: 'T-junction or triangle' },
  { char: '5', category: 'digit', expectedLoops: [0], expectedAspect: 0.65, notes: 'Top bar + bottom open lobe' },
  { char: '6', category: 'digit', expectedLoops: [1], expectedAspect: 0.65, bottomHeavy: true, notes: 'Bottom enclosed loop' },
  { char: '7', category: 'digit', expectedLoops: [0], expectedAspect: 0.6, topHeavy: true, notes: 'Top horizontal + diagonal' },
  { char: '8', category: 'digit', expectedLoops: [2], expectedAspect: 0.65, notes: 'Double enclosed loops' },
  { char: '9', category: 'digit', expectedLoops: [1], expectedAspect: 0.65, topHeavy: true, notes: 'Top enclosed loop with tail' },

  // Key Letters
  { char: 'A', category: 'uppercase', expectedLoops: [1], expectedAspect: 0.8, notes: 'Top apex, crossbar, upper triangular loop' },
  { char: 'B', category: 'uppercase', expectedLoops: [2], expectedAspect: 0.65, notes: 'Vertical stem + two enclosed loops' },
  { char: 'C', category: 'uppercase', expectedLoops: [0], expectedAspect: 0.7, notes: 'Open right curved arc' },
  { char: 'D', category: 'uppercase', expectedLoops: [1], expectedAspect: 0.75, notes: 'Vertical spine + single large loop' },
  { char: 'E', category: 'uppercase', expectedLoops: [0], expectedAspect: 0.6, notes: 'Vertical spine + 3 horizontal arms' },
  { char: 'F', category: 'uppercase', expectedLoops: [0], expectedAspect: 0.6, topHeavy: true, notes: 'Vertical spine + 2 upper arms' },
  { char: 'H', category: 'uppercase', expectedLoops: [0], expectedAspect: 0.75, typicalCrosses: 2, notes: 'Two parallel verticals + crossbar' },
  { char: 'I', category: 'uppercase', expectedLoops: [0], expectedAspect: 0.35, notes: 'Straight vertical stroke' },
  { char: 'L', category: 'uppercase', expectedLoops: [0], expectedAspect: 0.6, bottomHeavy: true, notes: 'Vertical stroke + horizontal floor' },
  { char: 'M', category: 'uppercase', expectedLoops: [0], expectedAspect: 0.9, notes: 'Wide symmetrical M peaks' },
  { char: 'O', category: 'uppercase', expectedLoops: [1], expectedAspect: 0.85, notes: 'Enclosed circular/oval loop' },
  { char: 'P', category: 'uppercase', expectedLoops: [1], expectedAspect: 0.65, topHeavy: true, notes: 'Vertical stem + upper loop' },
  { char: 'Q', category: 'uppercase', expectedLoops: [1], expectedAspect: 0.85, notes: 'Enclosed oval loop + tail' },
  { char: 'R', category: 'uppercase', expectedLoops: [1], expectedAspect: 0.7, notes: 'Upper loop + diagonal leg' },
  { char: 'S', category: 'uppercase', expectedLoops: [0], expectedAspect: 0.6, notes: 'Double smooth serpentine curve' },
  { char: 'T', category: 'uppercase', expectedLoops: [0], expectedAspect: 0.75, topHeavy: true, notes: 'Horizontal roof + central vertical' },
  { char: 'U', category: 'uppercase', expectedLoops: [0], expectedAspect: 0.7, bottomHeavy: true, notes: 'Open top cup' },
  { char: 'V', category: 'uppercase', expectedLoops: [0], expectedAspect: 0.75, bottomHeavy: true, notes: 'V-shaped diagonal valley' },
  { char: 'W', category: 'uppercase', expectedLoops: [0], expectedAspect: 0.95, notes: 'Wide double valley' },
  { char: 'X', category: 'uppercase', expectedLoops: [0], expectedAspect: 0.75, typicalCrosses: 1, notes: 'Diagonal intersection cross' },
  { char: 'Y', category: 'uppercase', expectedLoops: [0], expectedAspect: 0.7, topHeavy: true, notes: 'Upper V + lower stem' },
  { char: 'Z', category: 'uppercase', expectedLoops: [0], expectedAspect: 0.65, notes: 'Top horizontal + diagonal + bottom floor' },
];

/**
 * Score how well extracted topological & geometric features match character prototypes
 */
export function classifyAlphanumericByFeatures(
  features: AlphanumericFeatureBreakdown,
  pixels: Float32Array
): Array<{ char: string; confidence: number }> {
  const scores: Array<{ char: string; score: number }> = [];

  for (const proto of CHAR_PROTOTYPES) {
    let score = 1.0;

    // 1. Loop penalty / bonus (crucial topological invariant)
    if (proto.expectedLoops.includes(features.loopCount)) {
      score += 0.8;
    } else {
      score -= Math.abs(features.loopCount - proto.expectedLoops[0]) * 0.9;
    }

    // 2. Aspect ratio match
    const aspectDiff = Math.abs(features.aspectRatio - proto.expectedAspect);
    score -= aspectDiff * 0.5;

    // 3. Top-heavy vs bottom-heavy
    if (proto.topHeavy) {
      if (features.densityTop > 0.55) score += 0.4;
      else score -= 0.3;
    }
    if (proto.bottomHeavy) {
      if (features.densityBottom > 0.55) score += 0.4;
      else score -= 0.3;
    }

    // 4. Ambiguity dis-ambiguation rules
    // '0' vs 'O': '0' is typically taller/slender (aspect < 0.75), 'O' is wider (aspect >= 0.8)
    if (proto.char === '0' && features.aspectRatio > 0.82) score -= 0.25;
    if (proto.char === 'O' && features.aspectRatio < 0.65) score -= 0.25;

    // '1' vs 'I' vs 'l'
    if (proto.char === '1' && features.aspectRatio > 0.6) score -= 0.4;

    // '8' vs 'B': 'B' has flat vertical left spine; '8' has curved left indent
    if (proto.char === 'B' && features.densityLeft < features.densityRight) score += 0.3;
    if (proto.char === '8' && features.horizontalSymmetry > 0.6) score += 0.3;

    // '5' vs 'S': '5' has sharp top-horizontal bar and top-left corner
    if (proto.char === '5' && features.densityTop > 0.45 && features.densityBottom > 0.45) score += 0.2;

    scores.push({ char: proto.char, score: Math.max(0.01, score) });
  }

  // Sort descending by score
  scores.sort((a, b) => b.score - a.score);

  // Softmax normalization into percentage confidence
  const topCandidates = scores.slice(0, 5);
  const maxScore = topCandidates[0]?.score || 1;
  let expSum = 0;
  const expVals = topCandidates.map((c) => {
    const ev = Math.exp((c.score - maxScore) * 1.5);
    expSum += ev;
    return { char: c.char, ev };
  });

  return expVals.map((item) => ({
    char: item.char,
    confidence: item.ev / expSum,
  }));
}

/**
 * Character Segmentation via Connected Component Labeling on ink strokes:
 * Identifies distinct characters, words, and symbols across horizontal canvas
 */
export function segmentCharactersFromCanvas(
  sourceCanvas: HTMLCanvasElement
): SegmentedCharacter[] {
  const ctx = sourceCanvas.getContext('2d', { willReadFrequently: true });
  if (!ctx) return [];

  const w = sourceCanvas.width;
  const h = sourceCanvas.height;
  const imgData = ctx.getImageData(0, 0, w, h);
  const data = imgData.data;

  // Binary mask
  const isInk = new Uint8Array(w * h);
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      const idx = (y * w + x) * 4;
      const r = data[idx];
      const g = data[idx + 1];
      const b = data[idx + 2];
      const a = data[idx + 3] / 255;
      const darkness = 1 - (0.299 * r + 0.587 * g + 0.114 * b) / 255;
      if (darkness * a > 0.18) {
        isInk[y * w + x] = 1;
      }
    }
  }

  // Connected components via union-find or 2-pass BFS
  const visited = new Uint8Array(w * h);
  const boxes: Array<{ minX: number; maxX: number; minY: number; maxY: number; pixelCount: number }> = [];

  for (let y = 0; y < h; y += 2) {
    for (let x = 0; x < w; x += 2) {
      const idx = y * w + x;
      if (isInk[idx] && !visited[idx]) {
        // Found new cluster
        let minX = x,
          maxX = x,
          minY = y,
          maxY = y;
        let pixelCount = 0;

        const queue: [number, number][] = [[x, y]];
        visited[idx] = 1;

        while (queue.length > 0) {
          const [cx, cy] = queue.shift()!;
          pixelCount++;
          if (cx < minX) minX = cx;
          if (cx > maxX) maxX = cx;
          if (cy < minY) minY = cy;
          if (cy > maxY) maxY = cy;

          // Search neighbors (step size 2 for speed)
          const step = 2;
          for (let dy = -step; dy <= step; dy += step) {
            for (let dx = -step; dx <= step; dx += step) {
              const nx = cx + dx;
              const ny = cy + dy;
              if (nx >= 0 && nx < w && ny >= 0 && ny < h) {
                const nIdx = ny * w + nx;
                if (isInk[nIdx] && !visited[nIdx]) {
                  visited[nIdx] = 1;
                  queue.push([nx, ny]);
                }
              }
            }
          }
        }

        // Filter tiny flecks and stray dots (unless deliberate punctuation)
        if (pixelCount >= 10 && maxX - minX >= 4 && maxY - minY >= 8) {
          boxes.push({ minX, maxX, minY, maxY, pixelCount });
        }
      }
    }
  }

  if (boxes.length === 0) return [];

  // Merge overlapping or vertically stacked components belonging to same character (e.g. '=' or 'i' dot)
  const mergedBoxes: Array<{ minX: number; maxX: number; minY: number; maxY: number }> = [];

  for (const box of boxes) {
    let merged = false;
    for (const mb of mergedBoxes) {
      // If horizontal overlap is significant (e.g. 'i' dot directly over stem, or '=' bars)
      const hOverlap = Math.min(box.maxX, mb.maxX) - Math.max(box.minX, mb.minX);
      const minW = Math.min(box.maxX - box.minX, mb.maxX - mb.minX);
      const vDistance = Math.min(
        Math.abs(box.minY - mb.maxY),
        Math.abs(mb.minY - box.maxY)
      );

      if (hOverlap > 0.4 * minW && vDistance < 35) {
        mb.minX = Math.min(mb.minX, box.minX);
        mb.maxX = Math.max(mb.maxX, box.maxX);
        mb.minY = Math.min(mb.minY, box.minY);
        mb.maxY = Math.max(mb.maxY, box.maxY);
        merged = true;
        break;
      }
    }
    if (!merged) {
      mergedBoxes.push({ ...box });
    }
  }

  // Sort horizontally (Left-to-Right reading order)
  mergedBoxes.sort((a, b) => a.minX - b.minX);

  // Process each box into a SegmentedCharacter
  const results: SegmentedCharacter[] = [];

  for (let i = 0; i < mergedBoxes.length; i++) {
    const mb = mergedBoxes[i];
    const bw = mb.maxX - mb.minX + 1;
    const bh = mb.maxY - mb.minY + 1;

    // Crop and scale to 28x28
    const tempCanvas = document.createElement('canvas');
    tempCanvas.width = 28;
    tempCanvas.height = 28;
    const tctx = tempCanvas.getContext('2d')!;
    tctx.fillStyle = '#ffffff';
    tctx.fillRect(0, 0, 28, 28);

    const scale = 20 / Math.max(bw, bh);
    const tw = Math.max(1, Math.round(bw * scale));
    const th = Math.max(1, Math.round(bh * scale));
    const tx = Math.round((28 - tw) / 2);
    const ty = Math.round((28 - th) / 2);

    tctx.drawImage(sourceCanvas, mb.minX, mb.minY, bw, bh, tx, ty, tw, th);
    const tData = tctx.getImageData(0, 0, 28, 28).data;

    const charPixels = new Float32Array(784);
    for (let p = 0; p < 784; p++) {
      const r = tData[p * 4];
      const g = tData[p * 4 + 1];
      const b = tData[p * 4 + 2];
      const darkness = 1 - (0.299 * r + 0.587 * g + 0.114 * b) / 255;
      charPixels[p] = darkness;
    }

    const features = extractAlphanumericFeatures(charPixels);
    const candidates = classifyAlphanumericByFeatures(features, charPixels);
    const topChar = candidates[0]?.char || '?';

    results.push({
      id: `char-seg-${i}`,
      box: {
        x: mb.minX,
        y: mb.minY,
        width: bw,
        height: bh,
      },
      pixels: charPixels,
      features,
      predictedChar: topChar,
      candidates,
    });
  }

  return results;
}
