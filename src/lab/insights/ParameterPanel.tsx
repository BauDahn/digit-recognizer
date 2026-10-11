import React, { useMemo } from 'react';
import { NeuralNetwork } from '../../ml/neuralNetwork';
import { LayerSummary, summarizeNetwork } from '../../ml/netStats';

/** Weight histogram: how the values of one layer are spread, from -range to +range. */
const Histogram: React.FC<{ layer: LayerSummary }> = ({ layer }) => {
  const max = Math.max(1, ...layer.histogram);
  const w = 120;
  const h = 34;
  const bar = w / layer.histogram.length;
  return (
    <svg viewBox={`0 0 ${w} ${h + 10}`} className="w-full h-12" role="img" aria-label={`Histogram of the ${layer.name} weights`}>
      {layer.histogram.map((count, i) => (
        <rect key={i} x={i * bar} y={h - (count / max) * h} width={bar - 0.6} height={(count / max) * h} fill={i < layer.histogram.length / 2 ? '#fb7185' : '#60a5fa'} />
      ))}
      <line x1={w / 2} x2={w / 2} y1={0} y2={h} stroke="#94a3b8" strokeDasharray="2 2" strokeWidth="0.6" />
      <text x={0} y={h + 9} fontSize="7" fill="#94a3b8">{(-layer.histogramRange).toFixed(2)}</text>
      <text x={w / 2} y={h + 9} fontSize="7" fill="#94a3b8" textAnchor="middle">0</text>
      <text x={w} y={h + 9} fontSize="7" fill="#94a3b8" textAnchor="end">+{layer.histogramRange.toFixed(2)}</text>
    </svg>
  );
};

const fmt = (v: number, digits = 3) => (Math.abs(v) >= 100 ? v.toFixed(0) : v.toFixed(digits));
const bytes = (n: number) => (n > 1024 * 1024 ? `${(n / 1024 / 1024).toFixed(1)} MB` : `${(n / 1024).toFixed(0)} KB`);

interface Props {
  network: NeuralNetwork;
  version: number;
  /** Hyperparameters used for training (shown next to the architecture) */
  hyper?: { learningRate: number; momentum: number; batchSize: number; epochs?: number; augmentation?: boolean };
}

export const ParameterPanel: React.FC<Props> = ({ network, version, hyper }) => {
  const summary = useMemo(() => {
    void version;
    return summarizeNetwork(network);
  }, [network, version]);

  return (
    <div className="bg-white border border-slate-200 rounded-xl p-5 shadow-xs">
      <div className="flex flex-wrap items-baseline justify-between gap-2 mb-3">
        <span className="text-xs font-semibold uppercase tracking-wider text-slate-500">Parameters</span>
        <span className="text-xs text-slate-500 font-mono">
          {summary.totalParams.toLocaleString()} parameters · {bytes(summary.bytes)} · {summary.macs.toLocaleString()} multiply-adds per image
        </span>
      </div>

      <div className="overflow-x-auto">
        <table className="w-full text-xs">
          <thead>
            <tr className="text-left text-[10px] uppercase tracking-wider text-slate-400 border-b border-slate-100">
              <th className="py-1.5 pr-3 font-semibold">Layer</th>
              <th className="py-1.5 pr-3 font-semibold">Shape</th>
              <th className="py-1.5 pr-3 font-semibold text-right">Weights</th>
              <th className="py-1.5 pr-3 font-semibold text-right">Biases</th>
              <th className="py-1.5 pr-3 font-semibold text-right">Mean</th>
              <th className="py-1.5 pr-3 font-semibold text-right">Std</th>
              <th className="py-1.5 pr-3 font-semibold text-right">Min / max</th>
              <th className="py-1.5 pr-3 font-semibold text-right">‖W‖</th>
              <th className="py-1.5 pr-3 font-semibold text-right">≈ 0</th>
              <th className="py-1.5 font-semibold w-32">Distribution</th>
            </tr>
          </thead>
          <tbody className="font-mono text-slate-700">
            {summary.layers.map((l) => (
              <tr key={l.name} className="border-b border-slate-50 align-middle">
                <td className="py-1.5 pr-3 font-sans font-medium text-slate-900">{l.name}</td>
                <td className="py-1.5 pr-3">{l.inputs} → {l.outputs}</td>
                <td className="py-1.5 pr-3 text-right">{l.weightCount.toLocaleString()}</td>
                <td className="py-1.5 pr-3 text-right">{l.biasCount}</td>
                <td className="py-1.5 pr-3 text-right">{fmt(l.mean)}</td>
                <td className="py-1.5 pr-3 text-right">{fmt(l.std)}</td>
                <td className="py-1.5 pr-3 text-right">{fmt(l.min, 2)} / {fmt(l.max, 2)}</td>
                <td className="py-1.5 pr-3 text-right">{fmt(l.l2, 2)}</td>
                <td className="py-1.5 pr-3 text-right">{(l.nearZero * 100).toFixed(0)}%</td>
                <td className="py-1.5"><Histogram layer={l} /></td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      <dl className="mt-4 grid grid-cols-2 sm:grid-cols-4 gap-x-6 gap-y-2 text-xs">
        {[
          ['Activation', 'Leaky ReLU (slope 0.01)'],
          ['Output', 'Softmax, cross-entropy loss'],
          ['Initialization', 'He (std = √(2 / fan-in))'],
          ['Optimizer', hyper ? `SGD, momentum ${hyper.momentum}` : 'SGD with momentum'],
          ...(hyper ? [['Learning rate', String(hyper.learningRate)], ['Batch size', String(hyper.batchSize)], ['Data augmentation', hyper.augmentation ? 'on' : 'off']] : []),
        ].map(([k, v]) => (
          <div key={k}>
            <dt className="text-[10px] uppercase tracking-wider text-slate-400">{k}</dt>
            <dd className="text-slate-700">{v}</dd>
          </div>
        ))}
      </dl>
    </div>
  );
};
