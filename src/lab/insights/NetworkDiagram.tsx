import React, { useMemo, useState } from 'react';
import { NeuralNetwork } from '../../ml/neuralNetwork';
import type { NetworkWeights } from '../../ml/trainingClient';

/**
 * The network drawn as a graph: the input image, the two hidden layers and the output neurons.
 * Neurons are shaded by how much they fire for the current drawing, connections are blue when
 * positive and red when negative (thickness = strength). In "changes" mode the connections that
 * moved the most since the last update glow, so the learning can be watched.
 */

const W = 980;
const H = 470;
const COLS = { input: 70, h1: 330, h2: 600, out: 880 };

const MAX_H1 = 18;
const MAX_H2 = 14;
const MAX_OUT = 12;
const BLUE = '#2563eb';
const RED = '#e11d48';

interface NodeRef {
  layer: 1 | 2 | 3;
  index: number;
}

function pickIndices(count: number, limit: number, score: ArrayLike<number>): number[] {
  if (count <= limit) return Array.from({ length: count }, (_, i) => i);
  return Array.from({ length: count }, (_, i) => i)
    .sort((a, b) => score[b] - score[a])
    .slice(0, limit)
    .sort((a, b) => a - b);
}

function toDataUrl(width: number, height: number, fill: (i: number) => [number, number, number]): string {
  const canvas = document.createElement('canvas');
  canvas.width = width;
  canvas.height = height;
  const ctx = canvas.getContext('2d');
  if (!ctx) return '';
  const img = ctx.createImageData(width, height);
  for (let i = 0; i < width * height; i++) {
    const [r, g, b] = fill(i);
    img.data[i * 4] = r;
    img.data[i * 4 + 1] = g;
    img.data[i * 4 + 2] = b;
    img.data[i * 4 + 3] = 255;
  }
  ctx.putImageData(img, 0, 0);
  return canvas.toDataURL();
}

/** negative = red, positive = blue, zero = white */
function diverging(v: number, scale: number): [number, number, number] {
  const t = Math.max(-1, Math.min(1, v / scale));
  return t >= 0 ? [255 - 190 * t, 255 - 150 * t, 255 - 40 * t] : [255 - 20 * -t, 255 - 190 * -t, 255 - 160 * -t];
}

interface Props {
  network: NeuralNetwork;
  classes: string[];
  /** the current drawing (28x28), or null */
  pixels: Float32Array | null;
  /** the weights before the last update, to show what changed */
  previous: NetworkWeights | null;
  /** changes whenever the weights change */
  version: number;
  training?: boolean;
}

export const NetworkDiagram: React.FC<Props> = ({ network, classes, pixels, previous, version, training }) => {
  const [mode, setMode] = useState<'weights' | 'changes'>('weights');
  const [selected, setSelected] = useState<NodeRef | null>(null);
  const { hiddenSize1: n1, hiddenSize2: n2, outputSize: n3, inputSize } = network.config;

  const hasInput = !!pixels && pixels.some((v) => v > 0);

  // Forward pass for the current drawing (all neurons, then we pick the ones to draw)
  const pass = useMemo(() => {
    void version;
    return hasInput ? network.forward(pixels!) : null;
  }, [network, pixels, hasInput, version]);

  const norms1 = useMemo(() => {
    void version;
    const out = new Float32Array(n1);
    for (let i = 0; i < n1; i++) {
      let s = 0;
      for (let j = 0; j < inputSize; j++) s += network.w1[i * inputSize + j] ** 2;
      out[i] = Math.sqrt(s);
    }
    return out;
  }, [network, n1, inputSize, version]);

  const shown = useMemo(() => {
    void version;
    const score1 = pass ? pass.h1Act : norms1;
    const score2 = pass ? pass.h2Act : new Float32Array(n2).map((_, i) => i * -1);
    const scoreOut = pass ? pass.probabilities : new Float32Array(n3).map((_, i) => -i);
    return {
      h1: pickIndices(n1, MAX_H1, score1),
      h2: pickIndices(n2, MAX_H2, score2),
      out: pickIndices(n3, MAX_OUT, scoreOut),
    };
  }, [pass, norms1, n1, n2, n3, version]);

  const y = (position: number, count: number, top = 40, bottom = H - 40) => (count === 1 ? (top + bottom) / 2 : top + (position / (count - 1)) * (bottom - top));

  const inputImage = useMemo(() => (pixels ? toDataUrl(28, 28, (i) => { const v = Math.round(255 * (1 - Math.min(1, pixels[i]))); return [v, v, v]; }) : ''), [pixels]);

  const fieldImages = useMemo(() => {
    void version;
    return shown.h1.map((idx) => {
      const row = network.w1.subarray(idx * inputSize, (idx + 1) * inputSize);
      let max = 1e-6;
      for (let i = 0; i < row.length; i++) max = Math.max(max, Math.abs(row[i]));
      return toDataUrl(28, 28, (i) => diverging(row[i], max));
    });
  }, [shown.h1, network, inputSize, version]);

  // Connections between the drawn neurons
  const edges = useMemo(() => {
    void version;
    const out: Array<{ key: string; x1: number; y1: number; x2: number; y2: number; w: number; delta: number; from: NodeRef; to: NodeRef }> = [];
    const add = (layer: 2 | 3, fromList: number[], toList: number[], fromX: number, toX: number, weights: Float32Array, prev: Float32Array | undefined, fromCount: number) => {
      toList.forEach((to, ti) => {
        fromList.forEach((from, fi) => {
          const idx = to * fromCount + from;
          out.push({
            key: `${layer}-${from}-${to}`,
            x1: fromX,
            y1: y(fi, fromList.length),
            x2: toX,
            y2: y(ti, toList.length),
            w: weights[idx],
            delta: prev ? Math.abs(weights[idx] - prev[idx]) : 0,
            from: { layer: (layer - 1) as 1 | 2, index: from },
            to: { layer, index: to },
          });
        });
      });
    };
    add(2, shown.h1, shown.h2, COLS.h1 + 14, COLS.h2 - 14, network.w2, previous?.w2, n1);
    add(3, shown.h2, shown.out, COLS.h2 + 14, COLS.out - 14, network.w3, previous?.w3, n2);
    return out;
  }, [shown, network, previous, n1, n2, version]);

  const maxW2 = Math.max(1e-6, ...edges.filter((e) => e.to.layer === 2).map((e) => Math.abs(e.w)));
  const maxW3 = Math.max(1e-6, ...edges.filter((e) => e.to.layer === 3).map((e) => Math.abs(e.w)));
  const maxDelta = Math.max(1e-9, ...edges.map((e) => e.delta));
  const hasChanges = !!previous && maxDelta > 1e-9;

  // Input -> hidden 1: how much each drawn neuron's receptive field is excited
  const act1 = pass?.h1Act;
  const maxAct1 = act1 ? Math.max(1e-6, ...shown.h1.map((i) => act1[i])) : 1;

  const connected = (e: { from: NodeRef; to: NodeRef }) =>
    !selected || (e.from.layer === selected.layer && e.from.index === selected.index) || (e.to.layer === selected.layer && e.to.index === selected.index);

  const nodeActivation = (layer: 1 | 2 | 3, index: number): number => {
    if (!pass) return 0;
    if (layer === 1) return Math.max(0, pass.h1Act[index]) / maxAct1;
    if (layer === 2) return Math.max(0, pass.h2Act[index]) / Math.max(1e-6, ...shown.h2.map((i) => pass.h2Act[i]));
    return pass.probabilities[index];
  };

  const details = selected && describe(network, pass, selected, classes);
  const predicted = pass ? classes[pass.probabilities.indexOf(Math.max(...pass.probabilities))] : null;

  return (
    <div className="bg-white border border-slate-200 rounded-xl p-5 shadow-xs">
      <div className="flex flex-wrap items-center justify-between gap-2 mb-3">
        <div>
          <span className="text-xs font-semibold uppercase tracking-wider text-slate-500">The network</span>
          <p className="text-[11px] text-slate-400 mt-0.5">
            {inputSize} → {n1} → {n2} → {n3}. Showing the {shown.h1.length} / {shown.h2.length} / {shown.out.length} neurons that matter most for what you drew; click one to follow its connections.
          </p>
        </div>
        <div className="flex items-center gap-2">
          {training && <span className="text-[11px] text-amber-600 font-medium animate-pulse">training: weights are changing</span>}
          <div className="flex p-0.5 bg-slate-100 rounded-lg text-xs">
            {(['weights', 'changes'] as const).map((m) => (
              <button key={m} onClick={() => setMode(m)} className={`px-2.5 py-1 rounded-md font-medium ${mode === m ? 'bg-white shadow-2xs text-slate-900' : 'text-slate-500 hover:text-slate-800'}`}>
                {m === 'weights' ? 'Weights' : 'What changed'}
              </button>
            ))}
          </div>
        </div>
      </div>

      <svg viewBox={`0 0 ${W} ${H}`} className="w-full h-auto" role="img" aria-label={`Neural network diagram, ${inputSize}-${n1}-${n2}-${n3}`} onClick={() => setSelected(null)}>
        {/* column titles */}
        {[
          [COLS.input, 'Input (28×28)'],
          [COLS.h1, `Hidden 1 (${n1})`],
          [COLS.h2, `Hidden 2 (${n2})`],
          [COLS.out, `Output (${n3})`],
        ].map(([x, label]) => (
          <text key={label as string} x={x as number} y={16} textAnchor="middle" fontSize="11" fill="#64748b" fontWeight="600">
            {label}
          </text>
        ))}

        {/* input image and fan towards hidden 1 */}
        {shown.h1.map((idx, i) => (
          <line
            key={`in-${idx}`}
            x1={COLS.input + 44}
            y1={H / 2}
            x2={COLS.h1 - 24}
            y2={y(i, shown.h1.length)}
            stroke={mode === 'changes' ? '#f59e0b' : '#64748b'}
            strokeWidth={mode === 'changes' ? 0.6 : 0.5 + 1.6 * (act1 ? Math.max(0, act1[idx]) / maxAct1 : 0.2)}
            opacity={selected && !(selected.layer === 1 && selected.index === idx) ? 0.08 : act1 ? 0.15 + 0.6 * (Math.max(0, act1[idx]) / maxAct1) : 0.18}
          />
        ))}
        {inputImage ? (
          <image href={inputImage} x={COLS.input - 44} y={H / 2 - 44} width={88} height={88} style={{ imageRendering: 'pixelated' }} />
        ) : (
          <rect x={COLS.input - 44} y={H / 2 - 44} width={88} height={88} fill="#f8fafc" stroke="#e2e8f0" />
        )}
        <rect x={COLS.input - 44} y={H / 2 - 44} width={88} height={88} fill="none" stroke="#cbd5e1" />
        {!hasInput && (
          <text x={COLS.input} y={H / 2 + 62} textAnchor="middle" fontSize="10" fill="#94a3b8">
            draw something to see it fire
          </text>
        )}

        {/* connections */}
        {edges.map((e) => {
          const scale = e.to.layer === 2 ? maxW2 : maxW3;
          const strength = Math.abs(e.w) / scale;
          const dim = !connected(e);
          if (mode === 'changes') {
            const d = hasChanges ? e.delta / maxDelta : 0;
            return <line key={e.key} x1={e.x1} y1={e.y1} x2={e.x2} y2={e.y2} stroke="#f59e0b" strokeWidth={0.3 + 2.4 * d} opacity={dim ? 0.03 : 0.05 + 0.85 * d * d} />;
          }
          return <line key={e.key} x1={e.x1} y1={e.y1} x2={e.x2} y2={e.y2} stroke={e.w >= 0 ? BLUE : RED} strokeWidth={0.3 + 1.9 * strength} opacity={dim ? 0.03 : 0.08 + 0.7 * strength} />;
        })}

        {/* hidden 1: receptive fields */}
        {shown.h1.map((idx, i) => {
          const cy = y(i, shown.h1.length);
          const a = nodeActivation(1, idx);
          const isSel = selected?.layer === 1 && selected.index === idx;
          return (
            <g key={`h1-${idx}`} onClick={(ev) => { ev.stopPropagation(); setSelected({ layer: 1, index: idx }); }} className="cursor-pointer">
              <title>{`Hidden 1, neuron ${idx}${pass ? `, activation ${pass.h1Act[idx].toFixed(2)}` : ''}`}</title>
              <rect x={COLS.h1 - 22} y={cy - 11} width={22} height={22} rx={3} fill="#fff" stroke={isSel ? '#4f46e5' : '#cbd5e1'} strokeWidth={isSel ? 2 : 1} />
              <image href={fieldImages[i]} x={COLS.h1 - 21} y={cy - 10} width={20} height={20} style={{ imageRendering: 'pixelated' }} />
              <circle cx={COLS.h1 + 14} cy={cy} r={7} fill={`rgba(15,23,42,${0.08 + 0.92 * a})`} stroke={isSel ? '#4f46e5' : '#475569'} strokeWidth={isSel ? 2 : 1} />
            </g>
          );
        })}

        {/* hidden 2 */}
        {shown.h2.map((idx, i) => {
          const cy = y(i, shown.h2.length);
          const a = nodeActivation(2, idx);
          const isSel = selected?.layer === 2 && selected.index === idx;
          return (
            <g key={`h2-${idx}`} onClick={(ev) => { ev.stopPropagation(); setSelected({ layer: 2, index: idx }); }} className="cursor-pointer">
              <title>{`Hidden 2, neuron ${idx}${pass ? `, activation ${pass.h2Act[idx].toFixed(2)}` : ''}`}</title>
              <circle cx={COLS.h2} cy={cy} r={11} fill={`rgba(15,23,42,${0.08 + 0.92 * a})`} stroke={isSel ? '#4f46e5' : '#475569'} strokeWidth={isSel ? 2.5 : 1} />
              <text x={COLS.h2 - 20} y={cy + 3} textAnchor="end" fontSize="8" fill="#94a3b8">{idx}</text>
            </g>
          );
        })}

        {/* output */}
        {shown.out.map((idx, i) => {
          const cy = y(i, shown.out.length);
          const p = pass ? pass.probabilities[idx] : 0;
          const isSel = selected?.layer === 3 && selected.index === idx;
          const isTop = predicted === classes[idx] && hasInput;
          return (
            <g key={`out-${idx}`} onClick={(ev) => { ev.stopPropagation(); setSelected({ layer: 3, index: idx }); }} className="cursor-pointer">
              <title>{`Output "${classes[idx]}"${pass ? `, probability ${(p * 100).toFixed(1)}%` : ''}`}</title>
              <circle cx={COLS.out} cy={cy} r={13} fill={`rgba(15,23,42,${0.08 + 0.92 * p})`} stroke={isSel ? '#4f46e5' : isTop ? '#059669' : '#475569'} strokeWidth={isSel || isTop ? 2.5 : 1} />
              <text x={COLS.out} y={cy + 4} textAnchor="middle" fontSize="12" fontFamily="serif" fill={p > 0.5 ? '#fff' : '#0f172a'}>{classes[idx]}</text>
              <text x={COLS.out + 20} y={cy + 3} fontSize="9" fill="#64748b" className="font-mono">{pass ? `${(p * 100).toFixed(0)}%` : ''}</text>
            </g>
          );
        })}
        {n3 > shown.out.length && (
          <text x={COLS.out} y={H - 10} textAnchor="middle" fontSize="10" fill="#94a3b8">
            …and {n3 - shown.out.length} more outputs
          </text>
        )}
        {n1 > shown.h1.length && <text x={COLS.h1} y={H - 10} textAnchor="middle" fontSize="10" fill="#94a3b8">…and {n1 - shown.h1.length} more neurons</text>}
        {n2 > shown.h2.length && <text x={COLS.h2} y={H - 10} textAnchor="middle" fontSize="10" fill="#94a3b8">…and {n2 - shown.h2.length} more neurons</text>}
      </svg>

      <div className="flex flex-wrap items-center gap-x-5 gap-y-1 text-[11px] text-slate-500 mt-1">
        {mode === 'weights' ? (
          <>
            <span className="flex items-center gap-1.5"><span className="inline-block w-5 h-0.5" style={{ background: BLUE }} /> positive weight (excites)</span>
            <span className="flex items-center gap-1.5"><span className="inline-block w-5 h-0.5" style={{ background: RED }} /> negative weight (inhibits)</span>
          </>
        ) : (
          <span className="flex items-center gap-1.5">
            <span className="inline-block w-5 h-0.5 bg-amber-500" />
            {hasChanges ? 'connections that moved the most in the last update' : 'train the network to see which connections change'}
          </span>
        )}
        <span>Squares: what each first-layer neuron looks for (blue = ink there excites it). Shade of the circles: how much the neuron fires.</span>
      </div>

      {details && (
        <div className="mt-3 p-3 bg-slate-50 border border-slate-200 rounded-lg text-xs text-slate-700 grid grid-cols-2 sm:grid-cols-4 gap-3">
          <Detail label="Neuron" value={details.title} />
          <Detail label={details.layer === 3 ? 'Probability' : 'Activation'} value={details.activation} />
          <Detail label="Bias" value={details.bias} />
          <Detail label="Incoming weights" value={details.weights} />
          <div className="col-span-2 sm:col-span-4">
            <div className="text-[10px] uppercase tracking-wider text-slate-400 mb-1">{details.layer === 1 ? 'What it looks for' : 'Strongest inputs'}</div>
            <div className="font-mono text-[11px] text-slate-600">{details.top}</div>
          </div>
        </div>
      )}
    </div>
  );
};

const Detail: React.FC<{ label: string; value: string }> = ({ label, value }) => (
  <div>
    <div className="text-[10px] uppercase tracking-wider text-slate-400">{label}</div>
    <div className="font-mono">{value}</div>
  </div>
);

function describe(
  net: NeuralNetwork,
  pass: ReturnType<NeuralNetwork['forward']> | null,
  sel: NodeRef,
  classes: string[]
) {
  const { inputSize, hiddenSize1, hiddenSize2 } = net.config;
  const [w, b, fan] = sel.layer === 1 ? [net.w1, net.b1, inputSize] : sel.layer === 2 ? [net.w2, net.b2, hiddenSize1] : [net.w3, net.b3, hiddenSize2];
  const row = w.subarray(sel.index * fan, (sel.index + 1) * fan);
  let sum = 0;
  let sumSq = 0;
  for (const v of row) {
    sum += v;
    sumSq += v * v;
  }
  const activation = pass ? (sel.layer === 1 ? pass.h1Act[sel.index] : sel.layer === 2 ? pass.h2Act[sel.index] : pass.probabilities[sel.index]) : null;
  let top = '';
  if (sel.layer === 1) {
    const positive = Array.from(row).filter((v) => v > 0).length;
    top = `${positive} of ${fan} pixels excite it, ${fan - positive} inhibit it (see the square in the diagram)`;
  } else {
    const prevAct = pass ? (sel.layer === 2 ? pass.h1Act : pass.h2Act) : null;
    const list = Array.from(row, (v, i) => ({ i, v, contribution: prevAct ? v * prevAct[i] : v }))
      .sort((a, b2) => Math.abs(b2.contribution) - Math.abs(a.contribution))
      .slice(0, 5);
    top = list.map((c) => `#${c.i} ${c.v >= 0 ? '+' : ''}${c.v.toFixed(2)}`).join('   ');
  }
  return {
    layer: sel.layer,
    title: sel.layer === 3 ? `output “${classes[sel.index]}”` : `hidden ${sel.layer} · #${sel.index}`,
    activation: activation === null ? '—' : sel.layer === 3 ? `${(activation * 100).toFixed(1)}%` : activation.toFixed(3),
    bias: b[sel.index].toFixed(3),
    weights: `mean ${(sum / fan).toFixed(3)} · norm ${Math.sqrt(sumSq).toFixed(2)}`,
    top,
  };
}
