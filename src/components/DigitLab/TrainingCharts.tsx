import React, { useState } from 'react';
import { TrainingEpochMetric } from '../../types';

/**
 * Learning curves. Loss and accuracy have different scales, so they get two
 * separate small charts instead of one chart with two y-axes.
 */
export const TrainingCharts: React.FC<{ metrics: TrainingEpochMetric[] }> = ({ metrics }) => {
  const [hover, setHover] = useState<number | null>(null);
  if (metrics.length === 0) return null;

  const hovered = hover !== null ? metrics[hover] : metrics[metrics.length - 1];
  const maxLoss = Math.max(...metrics.map((m) => m.loss), 0.01);

  return (
    <div className="mt-3 space-y-2" onMouseLeave={() => setHover(null)}>
      <div className="flex justify-between text-[11px] text-slate-500">
        <span>Epoch {hovered.epoch}</span>
        <span className="font-mono">
          loss {hovered.loss.toFixed(3)} · test {(hovered.accuracy * 100).toFixed(1)}%
          {hovered.trainAccuracy !== undefined && ` · train ${(hovered.trainAccuracy * 100).toFixed(1)}%`}
        </span>
      </div>
      <MiniLineChart
        title="Training loss (lower is better)"
        values={metrics.map((m) => m.loss)}
        min={0}
        max={maxLoss}
        format={(v) => v.toFixed(2)}
        hover={hover}
        onHover={setHover}
      />
      <MiniLineChart
        title="Test accuracy (higher is better)"
        values={metrics.map((m) => m.accuracy)}
        min={Math.min(0.8, ...metrics.map((m) => m.accuracy))}
        max={1}
        format={(v) => `${(v * 100).toFixed(0)}%`}
        hover={hover}
        onHover={setHover}
      />
    </div>
  );
};

const W = 280;
const H = 64;
const PAD_L = 30;
const PAD_R = 6;
const PAD_Y = 6;

const MiniLineChart: React.FC<{
  title: string;
  values: number[];
  min: number;
  max: number;
  format: (v: number) => string;
  hover: number | null;
  onHover: (i: number | null) => void;
}> = ({ title, values, min, max, format, hover, onHover }) => {
  const n = values.length;
  const x = (i: number) => PAD_L + (n === 1 ? (W - PAD_L - PAD_R) / 2 : (i / (n - 1)) * (W - PAD_L - PAD_R));
  const y = (v: number) => PAD_Y + (1 - (v - min) / Math.max(1e-9, max - min)) * (H - 2 * PAD_Y);
  const points = values.map((v, i) => `${x(i)},${y(v)}`).join(' ');

  const handleMove = (e: React.MouseEvent<SVGSVGElement>) => {
    const rect = e.currentTarget.getBoundingClientRect();
    const px = ((e.clientX - rect.left) / rect.width) * W;
    const i = n === 1 ? 0 : Math.round(((px - PAD_L) / (W - PAD_L - PAD_R)) * (n - 1));
    onHover(Math.max(0, Math.min(n - 1, i)));
  };

  return (
    <div>
      <div className="text-[11px] text-slate-500 mb-0.5">{title}</div>
      <svg
        viewBox={`0 0 ${W} ${H}`}
        className="w-full h-16 bg-slate-50 rounded border border-slate-100"
        onMouseMove={handleMove}
        role="img"
        aria-label={`${title}: ${values.map(format).join(', ')}`}
      >
        {/* recessive axis labels: max and min */}
        <line x1={PAD_L} x2={W - PAD_R} y1={y(max)} y2={y(max)} stroke="#e2e8f0" strokeWidth="1" />
        <line x1={PAD_L} x2={W - PAD_R} y1={y(min)} y2={y(min)} stroke="#e2e8f0" strokeWidth="1" />
        <text x={PAD_L - 4} y={y(max) + 3} textAnchor="end" fontSize="8" fill="#94a3b8">
          {format(max)}
        </text>
        <text x={PAD_L - 4} y={y(min) + 3} textAnchor="end" fontSize="8" fill="#94a3b8">
          {format(min)}
        </text>
        {hover !== null && (
          <line x1={x(hover)} x2={x(hover)} y1={PAD_Y} y2={H - PAD_Y} stroke="#94a3b8" strokeWidth="1" strokeDasharray="2 2" />
        )}
        <polyline fill="none" stroke="#0f172a" strokeWidth="2" strokeLinejoin="round" strokeLinecap="round" points={points} />
        {(hover !== null ? [hover] : [n - 1]).map((i) => (
          <circle key={i} cx={x(i)} cy={y(values[i])} r="4" fill="#0f172a" stroke="#f8fafc" strokeWidth="2" />
        ))}
      </svg>
    </div>
  );
};
