import React from 'react';
import { LayerChange } from '../../ml/netStats';

export interface DynamicsPoint {
  /** position on the x axis (epochs, fractional while an epoch is running) */
  at: number;
  changes: LayerChange[];
}

const COLORS = ['#0f172a', '#2563eb', '#d97706'];
const NAMES = ['Hidden 1', 'Hidden 2', 'Output'];

/**
 * How far the weights of every layer moved at each update (mean |Δw|). Large at the start,
 * shrinking when the network settles.
 */
export const WeightDynamics: React.FC<{ points: DynamicsPoint[] }> = ({ points }) => {
  const W = 420;
  const H = 150;
  const pad = { l: 44, r: 10, t: 10, b: 20 };

  if (points.length < 2) {
    return (
      <div className="bg-white border border-slate-200 rounded-xl p-5 shadow-xs">
        <span className="text-xs font-semibold uppercase tracking-wider text-slate-500">How the weights move</span>
        <p className="text-xs text-slate-400 mt-2">Train the network: this chart shows how much each layer's weights change at every update.</p>
      </div>
    );
  }

  const max = Math.max(1e-9, ...points.flatMap((p) => p.changes.map((c) => c.meanAbs)));
  const x0 = points[0].at;
  const x1 = points[points.length - 1].at;
  const x = (v: number) => pad.l + ((v - x0) / Math.max(1e-9, x1 - x0)) * (W - pad.l - pad.r);
  const y = (v: number) => pad.t + (1 - v / max) * (H - pad.t - pad.b);
  const last = points[points.length - 1];

  return (
    <div className="bg-white border border-slate-200 rounded-xl p-5 shadow-xs">
      <div className="flex items-baseline justify-between mb-2">
        <span className="text-xs font-semibold uppercase tracking-wider text-slate-500">How the weights move</span>
        <span className="text-[11px] text-slate-400">mean |Δw| per update</span>
      </div>
      <svg viewBox={`0 0 ${W} ${H}`} className="w-full h-auto" role="img" aria-label="Mean absolute weight change per layer">
        {[0, 0.5, 1].map((t) => (
          <g key={t}>
            <line x1={pad.l} x2={W - pad.r} y1={y(max * t)} y2={y(max * t)} stroke="#e2e8f0" />
            <text x={pad.l - 4} y={y(max * t) + 3} fontSize="8" textAnchor="end" fill="#94a3b8">{(max * t).toExponential(1)}</text>
          </g>
        ))}
        {NAMES.map((name, layer) => (
          <polyline
            key={name}
            fill="none"
            stroke={COLORS[layer]}
            strokeWidth="1.8"
            strokeLinejoin="round"
            points={points.map((p) => `${x(p.at)},${y(p.changes[layer].meanAbs)}`).join(' ')}
          />
        ))}
        <text x={pad.l} y={H - 4} fontSize="9" fill="#94a3b8">epoch {x0.toFixed(1)}</text>
        <text x={W - pad.r} y={H - 4} fontSize="9" fill="#94a3b8" textAnchor="end">epoch {x1.toFixed(1)}</text>
      </svg>
      <div className="flex flex-wrap gap-x-4 gap-y-1 text-[11px] text-slate-600 mt-1">
        {NAMES.map((name, layer) => (
          <span key={name} className="flex items-center gap-1.5">
            <span className="inline-block w-3 h-0.5" style={{ background: COLORS[layer] }} />
            {name}
            <span className="font-mono text-slate-400">{last.changes[layer].meanAbs.toExponential(1)} ({(last.changes[layer].relative * 100).toFixed(1)}%)</span>
          </span>
        ))}
      </div>
    </div>
  );
};
