import { StrokePoint } from './imageOps';

/**
 * Split strokes written on one page into lines of math, so that several expressions
 * written one under the other are recognized separately.
 *
 * The strokes are projected on the vertical axis: wherever there is an empty band taller than
 * `gapFactor` times a typical symbol height, a new line starts. Fractions and limits stay together
 * because the parts of one formula sit closer than a line gap.
 */
export function splitIntoLines(strokes: StrokePoint[][], gapFactor = 0.4): number[][] {
  const spans = strokes
    .map((points, index) => {
      let minY = Infinity;
      let maxY = -Infinity;
      for (const p of points) {
        minY = Math.min(minY, p.y);
        maxY = Math.max(maxY, p.y);
      }
      return { index, minY, maxY };
    })
    .filter((s) => Number.isFinite(s.minY))
    .sort((a, b) => a.minY - b.minY);
  if (spans.length === 0) return [];

  // Typical height: the median of the taller half of the strokes (flat strokes and dots do not count)
  const heights = spans.map((s) => s.maxY - s.minY).sort((a, b) => a - b);
  const typical = heights[Math.floor(heights.length * 0.75)] || heights[heights.length - 1] || 1;
  const gap = gapFactor * Math.max(typical, 1);

  const lines: number[][] = [];
  let current: number[] = [];
  let bottom = -Infinity;
  for (const s of spans) {
    if (current.length > 0 && s.minY - bottom > gap) {
      lines.push(current);
      current = [];
      bottom = -Infinity;
    }
    current.push(s.index);
    bottom = Math.max(bottom, s.maxY);
  }
  lines.push(current);
  return lines;
}
