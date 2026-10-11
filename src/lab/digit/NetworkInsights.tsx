import React, { useEffect, useMemo, useRef, useState } from 'react';
import { NeuralNetwork } from '../../ml/neuralNetwork';
import { augment28 } from '../../ml/imageOps';
import { RefreshCw } from 'lucide-react';

/**
 * Educational views of what happens inside the network for the current drawing:
 * the activation of every neuron, layer by layer, and how data augmentation
 * creates new training images from one example.
 */

const PixelImage: React.FC<{ pixels: Float32Array; className?: string; title?: string }> = ({ pixels, className, title }) => {
  const ref = useRef<HTMLCanvasElement | null>(null);
  useEffect(() => {
    const ctx = ref.current?.getContext('2d');
    if (!ctx) return;
    const img = ctx.createImageData(28, 28);
    for (let i = 0; i < 784; i++) {
      const v = Math.round(255 * pixels[i]);
      img.data[i * 4] = v;
      img.data[i * 4 + 1] = v;
      img.data[i * 4 + 2] = v;
      img.data[i * 4 + 3] = 255;
    }
    ctx.putImageData(img, 0, 0);
  }, [pixels]);
  return <canvas ref={ref} width={28} height={28} title={title} className={`[image-rendering:pixelated] ${className ?? ''}`} />;
};

const LayerRow: React.FC<{ label: string; values: ArrayLike<number>; labels?: string[]; highlight?: number }> = ({
  label,
  values,
  labels,
  highlight,
}) => {
  const arr = Array.from(values);
  const max = Math.max(1e-6, ...arr.map((v) => Math.abs(v)));
  return (
    <div>
      <div className="flex justify-between text-[11px] text-slate-500 mb-1">
        <span>{label}</span>
        <span className="font-mono">{arr.length} neurons</span>
      </div>
      <div className="flex flex-wrap gap-[2px]">
        {arr.map((v, i) => {
          const intensity = Math.max(0, v) / max;
          return (
            <div
              key={i}
              title={`${labels ? labels[i] : `neuron ${i}`}: ${v.toFixed(3)}`}
              className={`flex items-center justify-center text-[9px] font-mono rounded-[2px] ${
                labels ? 'w-6 h-6' : 'w-3 h-3'
              } ${highlight === i ? 'ring-2 ring-emerald-500' : ''}`}
              style={{
                backgroundColor: `rgba(15, 23, 42, ${0.06 + 0.94 * intensity})`,
                color: intensity > 0.5 ? '#fff' : '#334155',
              }}
            >
              {labels?.[i]}
            </div>
          );
        })}
      </div>
    </div>
  );
};

export const LayerActivations: React.FC<{ network: NeuralNetwork; pixels: Float32Array }> = ({ network, pixels }) => {
  const hasInk = useMemo(() => pixels.some((v) => v > 0.1), [pixels]);
  // Not memoized: the weights change in place while training, and a forward pass is cheap
  const result = hasInk ? network.forward(pixels) : null;

  return (
    <div className="bg-white border border-slate-200 rounded-xl p-5 shadow-xs space-y-3">
      <div>
        <h3 className="text-sm font-semibold text-slate-900">Inside the network</h3>
        <p className="text-xs text-slate-500">
          Your drawing flows left to right: 784 input pixels → {network.config.hiddenSize1} → {network.config.hiddenSize2} →{' '}
          10 outputs. Darker squares are neurons that fire more (Leaky ReLU), the output layer is the softmax probability.
        </p>
      </div>
      {result ? (
        <div className="flex gap-4 items-start">
          <div className="shrink-0 text-center">
            <div className="text-[11px] text-slate-500 mb-1">Input</div>
            <PixelImage pixels={pixels} className="w-16 h-16 rounded bg-slate-900" />
          </div>
          <div className="flex-1 space-y-3 min-w-0">
            <LayerRow label="Hidden layer 1" values={result.h1Act} />
            <LayerRow label="Hidden layer 2" values={result.h2Act} />
            <LayerRow
              label="Output (probabilities)"
              values={result.probabilities}
              labels={['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']}
              highlight={Array.from(result.probabilities).indexOf(Math.max(...result.probabilities))}
            />
          </div>
        </div>
      ) : (
        <div className="text-xs text-slate-400 py-6 text-center">Draw a digit to see the activations.</div>
      )}
    </div>
  );
};

export const AugmentationPreview: React.FC<{ pixels: Float32Array; enabled: boolean }> = ({ pixels, enabled }) => {
  const [seed, setSeed] = useState(0);
  const hasInk = useMemo(() => pixels.some((v) => v > 0.1), [pixels]);
  const variants = useMemo(
    () => (hasInk ? Array.from({ length: 8 }, () => augment28(pixels)) : []),
    [pixels, hasInk, seed]
  );

  return (
    <div className="bg-white border border-slate-200 rounded-xl p-5 shadow-xs space-y-3">
      <div className="flex items-start justify-between gap-2">
        <div>
          <h3 className="text-sm font-semibold text-slate-900">Data augmentation {enabled ? '(on)' : '(off)'}</h3>
          <p className="text-xs text-slate-500">
            During training, half of the images are randomly rotated (±10°), scaled (±10%), shifted (±2px) or made
            bolder. The network sees "new" digits every epoch and generalizes better to your handwriting.
          </p>
        </div>
        <button
          onClick={() => setSeed((s) => s + 1)}
          disabled={!hasInk}
          className="p-1.5 text-slate-500 hover:text-slate-900 rounded-md hover:bg-slate-100 disabled:opacity-40"
          title="Generate other variations"
        >
          <RefreshCw className="w-4 h-4" />
        </button>
      </div>
      {hasInk ? (
        <div className="flex flex-wrap gap-2">
          <div className="text-center">
            <PixelImage pixels={pixels} className="w-12 h-12 rounded ring-2 ring-slate-400 bg-slate-900" />
            <div className="text-[10px] text-slate-500 mt-0.5">original</div>
          </div>
          {variants.map((v, i) => (
            <PixelImage key={`${seed}-${i}`} pixels={v} className="w-12 h-12 rounded bg-slate-900" />
          ))}
        </div>
      ) : (
        <div className="text-xs text-slate-400 py-4 text-center">Draw a digit to see augmented variations.</div>
      )}
    </div>
  );
};
