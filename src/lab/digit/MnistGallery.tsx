import React, { useState, useEffect, useRef } from 'react';
import { loadMnistGallerySamples, MnistSampleInfo } from '../../ml/mnistLoader';
import { Database, Eye, ArrowRight } from 'lucide-react';

interface MnistGalleryProps {
  onSelectSample: (sample: MnistSampleInfo) => void;
}

export const MnistGallery: React.FC<MnistGalleryProps> = ({ onSelectSample }) => {
  const [samples, setSamples] = useState<MnistSampleInfo[]>([]);
  const [filterDigit, setFilterDigit] = useState<number | 'all'>('all');
  const [loadError, setLoadError] = useState<string | null>(null);

  useEffect(() => {
    loadMnistGallerySamples()
      .then(setSamples)
      .catch((e) => setLoadError(String(e?.message || e)));
  }, []);

  const filtered = filterDigit === 'all' ? samples : samples.filter((s) => s.label === filterDigit);

  return (
    <div className="bg-white border border-slate-200 rounded-xl p-5 shadow-xs space-y-4">
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3">
        <div className="flex items-center gap-2">
          <div className="w-8 h-8 rounded-lg bg-indigo-50 border border-indigo-200 flex items-center justify-center text-indigo-600">
            <Database className="w-4 h-4" />
          </div>
          <div>
            <h3 className="text-sm font-semibold text-slate-900">
              Official Real MNIST Dataset Browser (Yann LeCun / NIST)
            </h3>
            <p className="text-xs text-slate-500">
              Real 28×28 grayscale digits from the MNIST test split (never used for training)
            </p>
          </div>
        </div>

        {/* Filter buttons */}
        <div className="flex items-center gap-1 overflow-x-auto pb-1 sm:pb-0">
          <button
            onClick={() => setFilterDigit('all')}
            className={`px-2.5 py-1 text-xs rounded-md font-medium transition-colors ${
              filterDigit === 'all'
                ? 'bg-slate-900 text-white'
                : 'bg-slate-100 text-slate-600 hover:bg-slate-200'
            }`}
          >
            All
          </button>
          {[0, 1, 2, 3, 4, 5, 6, 7, 8, 9].map((d) => (
            <button
              key={d}
              onClick={() => setFilterDigit(d)}
              className={`w-7 h-7 text-xs font-mono font-medium rounded-md transition-colors ${
                filterDigit === d
                  ? 'bg-indigo-600 text-white font-bold'
                  : 'bg-slate-100 text-slate-700 hover:bg-slate-200'
              }`}
            >
              {d}
            </button>
          ))}
        </div>
      </div>

      {loadError && <div className="text-xs text-rose-600">Could not load MNIST: {loadError}</div>}

      {/* Grid of mini MNIST sample thumbnails */}
      <div className="grid grid-cols-6 sm:grid-cols-10 md:grid-cols-12 lg:grid-cols-15 gap-2 max-h-48 overflow-y-auto p-2 bg-slate-50 border border-slate-200 rounded-lg">
        {filtered.slice(0, 90).map((item) => (
          <MnistThumbnail
            key={item.id}
            sample={item}
            onClick={() => onSelectSample(item)}
          />
        ))}
      </div>

      <div className="flex items-center justify-between text-[11px] text-slate-500">
        <span>Click any real MNIST sample to load it directly into the drawing canvas & test the neural network!</span>
        <span className="font-mono">{filtered.length} samples shown</span>
      </div>
    </div>
  );
};

// Mini 28x28 canvas thumbnail for each MNIST sample
const MnistThumbnail: React.FC<{
  sample: MnistSampleInfo;
  onClick: () => void;
}> = ({ sample, onClick }) => {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const imgData = ctx.createImageData(28, 28);
    for (let i = 0; i < 784; i++) {
      // MNIST has dark background (0) and white ink (255)
      const v = Math.round(sample.pixels[i] * 255);
      imgData.data[i * 4] = v;
      imgData.data[i * 4 + 1] = v;
      imgData.data[i * 4 + 2] = v;
      imgData.data[i * 4 + 3] = 255;
    }
    ctx.putImageData(imgData, 0, 0);
  }, [sample]);

  return (
    <button
      onClick={onClick}
      className="group relative flex flex-col items-center bg-slate-900 rounded p-1 hover:ring-2 hover:ring-indigo-500 transition-all cursor-pointer"
      title={`Real MNIST sample: digit ${sample.label}`}
    >
      <canvas
        ref={canvasRef}
        width={28}
        height={28}
        className="w-9 h-9 image-rendering-pixelated rounded-xs"
      />
      <span className="text-[10px] font-mono font-bold text-slate-300 group-hover:text-white mt-0.5">
        {sample.label}
      </span>
    </button>
  );
};
