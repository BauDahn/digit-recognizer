import React, { useEffect, useRef } from 'react';
import { RecognizedSymbol, SymbolRole } from '../ml/mathLayout';
import { displayChar } from '../ml/strokeRecognizer';

const ROLE_LABEL: Partial<Record<SymbolRole, string>> = {
  sup: 'exponent',
  sub: 'index',
  numerator: 'numerator',
  denominator: 'denominator',
  radicand: 'under √',
  'root-index': 'root index',
  'upper-limit': 'upper limit',
  'lower-limit': 'lower limit',
};

/** The 28x28 image the network saw, drawn small. */
export const PixelPreview: React.FC<{ pixels: Float32Array; className?: string }> = ({ pixels, className }) => {
  const ref = useRef<HTMLCanvasElement | null>(null);
  useEffect(() => {
    const ctx = ref.current?.getContext('2d');
    if (!ctx) return;
    const img = ctx.createImageData(28, 28);
    for (let i = 0; i < 784; i++) {
      const v = Math.round(255 * (1 - pixels[i]));
      img.data[i * 4] = v;
      img.data[i * 4 + 1] = v;
      img.data[i * 4 + 2] = v;
      img.data[i * 4 + 3] = 255;
    }
    ctx.putImageData(img, 0, 0);
  }, [pixels]);
  return <canvas ref={ref} width={28} height={28} className={className ?? 'w-12 h-12 [image-rendering:pixelated]'} />;
};

interface SymbolTileProps {
  symbol: RecognizedSymbol;
  active?: boolean;
  onClick: () => void;
}

export const SymbolTile: React.FC<SymbolTileProps> = ({ symbol, active, onClick }) => {
  const low = symbol.confidence < 0.6 && !symbol.corrected;
  const role = ROLE_LABEL[symbol.role];
  return (
    <button
      type="button"
      onClick={onClick}
      title={symbol.candidates.map((c) => `${displayChar(c.char)}  ${(c.confidence * 100).toFixed(1)}%`).join('\n') + '\n\nClick to correct'}
      className={`relative w-[68px] p-1.5 rounded-lg border text-center transition-colors focus:outline-hidden focus-visible:ring-2 focus-visible:ring-indigo-400 ${
        active
          ? 'border-indigo-500 bg-indigo-50'
          : symbol.corrected
            ? 'border-emerald-300 bg-emerald-50 hover:bg-emerald-100'
            : low
              ? 'border-amber-300 bg-amber-50 hover:bg-amber-100'
              : 'border-slate-200 bg-white hover:bg-slate-50'
      }`}
    >
      {symbol.pixels ? (
        <PixelPreview pixels={symbol.pixels} className="w-12 h-12 mx-auto border border-slate-100 [image-rendering:pixelated]" />
      ) : (
        <div className="w-12 h-12 mx-auto flex items-center justify-center text-[10px] text-slate-400">by size</div>
      )}
      <div className="mt-1 font-serif text-lg leading-none text-slate-900">{displayChar(symbol.char)}</div>
      <div className="text-[10px] text-slate-500 mt-0.5">{symbol.corrected ? 'you said' : `${(symbol.confidence * 100).toFixed(0)}%`}</div>
      {role && <div className="text-[9px] text-indigo-500 truncate">{role}</div>}
    </button>
  );
};
