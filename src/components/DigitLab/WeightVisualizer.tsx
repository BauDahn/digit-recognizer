import React, { useEffect, useRef, useState } from 'react';

interface WeightVisualizerProps {
  weightsW1: Float32Array; // [hiddenSize1 x 784]
  hiddenSize1: number;
  activations?: number[];
}

export const WeightVisualizer: React.FC<WeightVisualizerProps> = ({
  weightsW1,
  hiddenSize1,
  activations,
}) => {
  const [selectedNeuron, setSelectedNeuron] = useState<number>(0);
  const canvasRef = useRef<HTMLCanvasElement | null>(null);

  // Render the selected neuron's 28x28 filter weights
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || !weightsW1 || weightsW1.length < 784) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const offset = selectedNeuron * 784;
    const imgData = ctx.createImageData(28, 28);

    // Find min and max weight for normalized bipolar colormap (blue = negative, white = 0, red = positive)
    let maxAbs = 0.001;
    for (let i = 0; i < 784; i++) {
      const val = Math.abs(weightsW1[offset + i]);
      if (val > maxAbs) maxAbs = val;
    }

    for (let i = 0; i < 784; i++) {
      const w = weightsW1[offset + i] / maxAbs; // -1 to +1
      let r = 255,
        g = 255,
        b = 255;

      if (w > 0) {
        // Red for positive excitation
        g = Math.round(255 * (1 - w));
        b = Math.round(255 * (1 - w));
      } else {
        // Blue for negative inhibition
        const nw = Math.abs(w);
        r = Math.round(255 * (1 - nw));
        g = Math.round(255 * (1 - nw));
      }

      imgData.data[i * 4] = r;
      imgData.data[i * 4 + 1] = g;
      imgData.data[i * 4 + 2] = b;
      imgData.data[i * 4 + 3] = 255;
    }

    ctx.putImageData(imgData, 0, 0);
  }, [weightsW1, selectedNeuron]);

  return (
    <div className="bg-white border border-slate-200 rounded-xl p-5 shadow-xs">
      <div className="flex items-center justify-between mb-4">
        <div>
          <h3 className="text-sm font-semibold text-slate-900">Neural Feature Detectors</h3>
          <p className="text-xs text-slate-500">
            Receptive field weights connecting 784 input pixels to Hidden Layer 1
          </p>
        </div>
        <div className="text-xs text-slate-500 font-mono">
          Neuron #{selectedNeuron + 1} of {hiddenSize1}
        </div>
      </div>

      <div className="flex flex-col sm:flex-row items-center gap-6">
        {/* Large magnified view of selected neuron */}
        <div className="flex flex-col items-center">
          <div className="p-2 bg-slate-900 rounded-lg shadow-inner">
            <canvas
              ref={canvasRef}
              width={28}
              height={28}
              className="w-32 h-32 image-rendering-pixelated rounded-xs border border-slate-700"
            />
          </div>
          <div className="flex items-center justify-center gap-3 mt-2 text-[11px] text-slate-500">
            <span className="flex items-center gap-1">
              <span className="w-2.5 h-2.5 rounded-full bg-blue-500"></span> Inhibitory (-)
            </span>
            <span className="flex items-center gap-1">
              <span className="w-2.5 h-2.5 rounded-full bg-red-500"></span> Excitatory (+)
            </span>
          </div>
          {activations && activations[selectedNeuron] !== undefined && (
            <div className="mt-2 text-xs font-mono text-slate-700 bg-slate-100 px-2 py-0.5 rounded">
              Current Activation: {activations[selectedNeuron].toFixed(3)}
            </div>
          )}
        </div>

        {/* Thumbnail selector of all hidden neurons */}
        <div className="flex-1 w-full">
          <label className="text-xs font-medium text-slate-600 block mb-1.5">
            Select Hidden Neuron to Inspect:
          </label>
          <div className="grid grid-cols-8 sm:grid-cols-12 gap-1.5 max-h-36 overflow-y-auto p-1.5 border border-slate-200 rounded-lg bg-slate-50">
            {Array.from({ length: Math.min(hiddenSize1, 64) }, (_, idx) => {
              const act = activations?.[idx] ?? 0;
              const isSelected = idx === selectedNeuron;
              return (
                <button
                  key={idx}
                  onClick={() => setSelectedNeuron(idx)}
                  title={`Neuron ${idx + 1}, Activation: ${act.toFixed(2)}`}
                  className={`w-7 h-7 text-[10px] font-mono rounded flex items-center justify-center transition-all ${
                    isSelected
                      ? 'bg-slate-900 text-white font-bold ring-2 ring-slate-400'
                      : act > 0.5
                      ? 'bg-amber-200 text-amber-900 border border-amber-300'
                      : 'bg-white text-slate-600 border border-slate-200 hover:bg-slate-100'
                  }`}
                >
                  {idx + 1}
                </button>
              );
            })}
          </div>
        </div>
      </div>
    </div>
  );
};
