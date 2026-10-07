import React, { useRef, useState, useEffect, useCallback } from 'react';
import {
  extractAlphanumericFeatures,
  classifyAlphanumericByFeatures,
  segmentCharactersFromCanvas,
} from '../../ml/alphanumericFeatures';
import { SegmentedCharacter, AlphanumericFeatureBreakdown } from '../../types';
import { Trash2, HelpCircle, Layers, Cpu, Compass, CheckCircle2 } from 'lucide-react';
import { LoadedModel, loadBuiltinModel, fromModelFile, loadModelFromStorage } from '../../ml/modelFile';
import { preprocessCanvasTo28x28 } from '../../ml/preprocessing';
import { LettersTrainer } from './LettersTrainer';

export const AlphanumericStudio: React.FC = () => {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const [isDrawing, setIsDrawing] = useState(false);
  const [brushSize, setBrushSize] = useState<number>(14);
  const [segmentedChars, setSegmentedChars] = useState<SegmentedCharacter[]>([]);
  const [selectedCharId, setSelectedCharId] = useState<string | null>(null);
  const [recognizedString, setRecognizedString] = useState<string>('');

  // Ambiguity lab active test pair
  const [activePairKey, setActivePairKey] = useState<'0_vs_O' | '1_vs_I' | '5_vs_S' | '8_vs_B' | '2_vs_Z'>('0_vs_O');

  // Neural network trained on EMNIST Balanced (47 classes). A model the user trained
  // in this browser takes precedence over the shipped pre-trained one.
  const [model, setModel] = useState<LoadedModel | null>(null);
  const [modelError, setModelError] = useState<string | null>(null);
  useEffect(() => {
    const saved = loadModelFromStorage('letters');
    if (saved) {
      try {
        setModel(fromModelFile(saved));
        return;
      } catch {
        // fall back to the pre-trained model
      }
    }
    loadBuiltinModel('letters')
      .then(setModel)
      .catch((e) => setModelError(String(e?.message || e)));
  }, []);

  // Initialize canvas
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    ctx.fillStyle = '#ffffff';
    ctx.fillRect(0, 0, canvas.width, canvas.height);
  }, []);

  // Run segmentation and recognition
  const processCanvas = useCallback(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const chars = segmentCharactersFromCanvas(canvas);

    // Classify every segmented character with the network too (same MNIST-style preprocessing)
    if (model) {
      for (const c of chars) {
        const { pixels } = preprocessCanvasTo28x28(canvas, c.box);
        const { probabilities } = model.network.forward(pixels);
        const ranked = model.classes
          .map((char, i) => ({ char, confidence: probabilities[i] }))
          .sort((a, b) => b.confidence - a.confidence)
          .slice(0, 5);
        c.nnPixels = pixels;
        c.nnCandidates = ranked;
        c.nnChar = ranked[0].char;
      }
    }
    setSegmentedChars(chars);

    const str = chars.map((c) => c.nnChar ?? c.predictedChar).join('');
    setRecognizedString(str);

    if (chars.length > 0 && !selectedCharId) {
      setSelectedCharId(chars[0].id);
    }
  }, [selectedCharId, model]);

  // Re-run recognition when the model changes (loaded or retrained)
  useEffect(() => {
    processCanvas();
  }, [model]);

  // Pointer event handlers
  const handlePointerDown = (e: React.PointerEvent<HTMLCanvasElement>) => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const rect = canvas.getBoundingClientRect();
    const scaleX = canvas.width / rect.width;
    const scaleY = canvas.height / rect.height;
    const x = (e.clientX - rect.left) * scaleX;
    const y = (e.clientY - rect.top) * scaleY;

    ctx.beginPath();
    ctx.moveTo(x, y);
    ctx.lineWidth = brushSize;
    ctx.lineCap = 'round';
    ctx.lineJoin = 'round';
    ctx.strokeStyle = '#0f172a';

    setIsDrawing(true);
    canvas.setPointerCapture(e.pointerId);
  };

  const handlePointerMove = (e: React.PointerEvent<HTMLCanvasElement>) => {
    if (!isDrawing) return;
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const rect = canvas.getBoundingClientRect();
    const scaleX = canvas.width / rect.width;
    const scaleY = canvas.height / rect.height;
    const x = (e.clientX - rect.left) * scaleX;
    const y = (e.clientY - rect.top) * scaleY;

    ctx.lineTo(x, y);
    ctx.stroke();
    processCanvas();
  };

  const handlePointerUp = (e: React.PointerEvent<HTMLCanvasElement>) => {
    if (!isDrawing) return;
    setIsDrawing(false);
    processCanvas();
    try {
      e.currentTarget.releasePointerCapture(e.pointerId);
    } catch {
      // ignore
    }
  };

  const handleClear = () => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    ctx.fillStyle = '#ffffff';
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    setSegmentedChars([]);
    setSelectedCharId(null);
    setRecognizedString('');
  };

  const selectedChar = segmentedChars.find((c) => c.id === selectedCharId) || segmentedChars[0];

  // Draw sample alphanumeric text into canvas
  const handleLoadSampleString = (text: string) => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    ctx.fillStyle = '#ffffff';
    ctx.fillRect(0, 0, canvas.width, canvas.height);

    ctx.fillStyle = '#0f172a';
    ctx.font = 'bold 80px "Inter", sans-serif';
    ctx.textBaseline = 'middle';
    ctx.textAlign = 'center';

    ctx.fillText(text, canvas.width / 2, canvas.height / 2);
    processCanvas();
  };

  const AMBIGUOUS_PAIRS = {
    '0_vs_O': {
      title: 'Digit "0" vs Letter "O"',
      difference:
        'Digit "0" is typically slender with aspect ratio < 0.75, while Letter "O" is wide and circular (aspect ratio > 0.82). Both have 1 Euler loop.',
      testA: '0',
      testB: 'O',
    },
    '1_vs_I': {
      title: 'Digit "1" vs Letter "I"',
      difference:
        'European "1" features a top diagonal hook. Capital "I" typically has horizontal serif bars at top and bottom, or pure vertical stroke.',
      testA: '1',
      testB: 'I',
    },
    '5_vs_S': {
      title: 'Digit "5" vs Letter "S"',
      difference:
        '"5" possesses a sharp 90° top-left corner and flat top bar. "S" consists of continuous smooth serpentine curvature with no hard corners.',
      testA: '5',
      testB: 'S',
    },
    '8_vs_B': {
      title: 'Digit "8" vs Letter "B"',
      difference:
        'Both have 2 Euler loops! However, "B" has a rigid straight vertical left spine, whereas "8" has bilateral symmetry with waist indents on both sides.',
      testA: '8',
      testB: 'B',
    },
    '2_vs_Z': {
      title: 'Digit "2" vs Letter "Z"',
      difference:
        '"2" has a smooth curved convex crown. "Z" consists exclusively of straight lines with two sharp acute corners.',
      testA: '2',
      testB: 'Z',
    },
  };

  return (
    <div className="max-w-7xl mx-auto px-4 sm:px-6 py-6 space-y-6">
      {/* Header */}
      <div className="flex flex-col md:flex-row md:items-center justify-between gap-4 pb-4 border-b border-slate-200">
        <div>
          <h1 className="text-xl font-bold text-slate-900 tracking-tight">
            Stage 2: Alphanumeric Studio (0–9, A–Z)
          </h1>
          <p className="text-xs text-slate-500 mt-1 max-w-2xl">
            Expand beyond digits to letters. Characters are segmented, then classified by a neural network
            trained on EMNIST (47 classes) and, for comparison, by hand-written rules based on loops and symmetry.
          </p>
          <p className="text-[11px] mt-1 text-slate-400">
            {model
              ? `Network: ${model.network.config.inputSize}-${model.network.config.hiddenSize1}-${model.network.config.hiddenSize2}-${model.network.config.outputSize}, ${model.meta?.trainedOn ?? 'EMNIST'}${model.meta?.testAccuracy ? `, ${(model.meta.testAccuracy * 100).toFixed(1)}% test accuracy` : ''}`
              : modelError
              ? `Network unavailable (${modelError}); using rules only.`
              : 'Loading network…'}
          </p>
        </div>

        {/* Quick sample buttons */}
        <div className="flex items-center gap-1.5 flex-wrap">
          <span className="text-xs text-slate-500 font-medium mr-1">Load Preset:</span>
          {['E=MC2', 'MATH99', '7A3B', 'F=MA'].map((txt) => (
            <button
              key={txt}
              onClick={() => handleLoadSampleString(txt)}
              className="px-2.5 py-1 text-xs font-mono font-medium rounded-md bg-white border border-slate-200 hover:bg-slate-100 text-slate-700 transition-colors shadow-2xs"
            >
              {txt}
            </button>
          ))}
        </div>
      </div>

      {/* Main Drawing & Multi-character Canvas */}
      <div className="bg-white border border-slate-200 rounded-xl p-5 shadow-xs space-y-4">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2">
            <span className="text-xs font-semibold uppercase tracking-wider text-slate-500">
              Multi-Character Canvas
            </span>
            <span className="text-xs text-slate-400">
              (Write multiple letters or digits separated by space)
            </span>
          </div>

          <div className="flex items-center gap-3">
            <label className="text-[11px] text-slate-500 flex items-center gap-1">
              Brush:
              <input
                type="range"
                min="8"
                max="24"
                value={brushSize}
                onChange={(e) => setBrushSize(Number(e.target.value))}
                className="w-16 h-1 bg-slate-200 rounded-lg appearance-none cursor-pointer"
              />
            </label>
            <button
              onClick={handleClear}
              className="p-1.5 text-slate-500 hover:text-rose-600 rounded-md hover:bg-slate-100 transition-colors"
              title="Clear canvas"
            >
              <Trash2 className="w-4 h-4" />
            </button>
          </div>
        </div>

        {/* Canvas & Overlay bounding boxes */}
        <div className="relative border border-slate-200 rounded-lg overflow-hidden bg-slate-50">
          <canvas
            ref={canvasRef}
            width={860}
            height={220}
            onPointerDown={handlePointerDown}
            onPointerMove={handlePointerMove}
            onPointerUp={handlePointerUp}
            className="w-full h-48 bg-white cursor-crosshair touch-none"
          />

          {/* Render bounding boxes over recognized characters */}
          {segmentedChars.map((sc) => {
            const isSelected = selectedChar?.id === sc.id;
            const canvasEl = canvasRef.current;
            if (!canvasEl) return null;
            const factorX = 100 / canvasEl.width;
            const factorY = 100 / canvasEl.height;

            return (
              <div
                key={sc.id}
                onClick={() => setSelectedCharId(sc.id)}
                className={`absolute border-2 rounded-xs transition-all cursor-pointer ${
                  isSelected
                    ? 'border-indigo-600 bg-indigo-500/10 shadow-xs'
                    : 'border-slate-400/80 hover:border-slate-700 bg-slate-400/5'
                }`}
                style={{
                  left: `${sc.box.x * factorX}%`,
                  top: `${sc.box.y * factorY}%`,
                  width: `${sc.box.width * factorX}%`,
                  height: `${sc.box.height * factorY}%`,
                }}
              >
                <span
                  className={`absolute -top-5 left-0 px-1 py-0.2 text-[10px] font-mono font-bold rounded-xs ${
                    isSelected ? 'bg-indigo-600 text-white' : 'bg-slate-700 text-white'
                  }`}
                >
                  {sc.nnChar ?? sc.predictedChar}
                </span>
              </div>
            );
          })}
        </div>

        {/* Recognized String Banner */}
        <div className="p-3 bg-slate-50 border border-slate-200 rounded-lg flex items-center justify-between">
          <div className="flex items-center gap-3">
            <span className="text-xs text-slate-500 uppercase tracking-wider font-semibold">
              Recognized Sequence:
            </span>
            <span className="text-lg font-mono font-bold text-slate-900 tracking-wider">
              {recognizedString || '— (Draw characters on canvas above)'}
            </span>
            {segmentedChars.length > 0 && model && (
              <span className="text-xs text-slate-400 font-mono">
                rules only: {segmentedChars.map((c) => c.predictedChar).join('')}
              </span>
            )}
          </div>

          <div className="text-xs text-slate-500 font-mono">
            {segmentedChars.length} character{segmentedChars.length === 1 ? '' : 's'} segmented
          </div>
        </div>
      </div>

      {/* Grid: Character Feature Breakdown + Ambiguity Inspector */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
        {/* Left: Detailed Topological Feature Breakdown of selected character */}
        <div className="lg:col-span-7 bg-white border border-slate-200 rounded-xl p-5 shadow-xs space-y-4">
          <div className="flex items-center justify-between">
            <div>
              <h3 className="text-sm font-semibold text-slate-900">
                Topological & Geometric Feature Inspector
              </h3>
              <p className="text-xs text-slate-500">
                Structural invariants used to classify alphanumeric characters
              </p>
            </div>
            {selectedChar && (
              <div className="flex items-center gap-1.5">
                <span className="text-xs text-slate-500">Selected:</span>
                <span className="w-7 h-7 flex items-center justify-center font-mono font-bold text-sm bg-slate-900 text-white rounded-md">
                  {selectedChar.nnChar ?? selectedChar.predictedChar}
                </span>
              </div>
            )}
          </div>

          {selectedChar ? (
            <div className="space-y-4">
              {/* Feature Grid */}
              <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 text-xs">
                <div className="p-2.5 bg-slate-50 rounded-lg border border-slate-100">
                  <span className="text-slate-400 block text-[11px]">Euler Loops (Holes)</span>
                  <span className="text-base font-mono font-bold text-slate-900">
                    {selectedChar.features.loopCount}
                  </span>
                  <span className="text-[10px] text-slate-500 block mt-0.5">
                    {selectedChar.features.loopCount === 2
                      ? '8, B'
                      : selectedChar.features.loopCount === 1
                      ? '0, 6, 9, A, D, O, P, Q, R'
                      : '1, 2, 3, 4, 5, 7, C, E, ...'}
                  </span>
                </div>

                <div className="p-2.5 bg-slate-50 rounded-lg border border-slate-100">
                  <span className="text-slate-400 block text-[11px]">Aspect Ratio (w/h)</span>
                  <span className="text-base font-mono font-bold text-slate-900">
                    {selectedChar.features.aspectRatio.toFixed(2)}
                  </span>
                  <span className="text-[10px] text-slate-500 block mt-0.5">
                    {selectedChar.features.aspectRatio < 0.5 ? 'Slender (1, I)' : 'Wide (M, W, O)'}
                  </span>
                </div>

                <div className="p-2.5 bg-slate-50 rounded-lg border border-slate-100">
                  <span className="text-slate-400 block text-[11px]">Horizontal Symmetry</span>
                  <span className="text-base font-mono font-bold text-slate-900">
                    {(selectedChar.features.horizontalSymmetry * 100).toFixed(0)}%
                  </span>
                  <span className="text-[10px] text-slate-500 block mt-0.5">Left-right mirror</span>
                </div>

                <div className="p-2.5 bg-slate-50 rounded-lg border border-slate-100">
                  <span className="text-slate-400 block text-[11px]">Vertical Symmetry</span>
                  <span className="text-base font-mono font-bold text-slate-900">
                    {(selectedChar.features.verticalSymmetry * 100).toFixed(0)}%
                  </span>
                  <span className="text-[10px] text-slate-500 block mt-0.5">Top-bottom mirror</span>
                </div>
              </div>

              {/* Network vs. rules, side by side */}
              <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
                <div>
                  <span className="text-xs font-semibold text-slate-700 flex items-center gap-2 mb-2">
                    Neural network (EMNIST)
                    {selectedChar.nnPixels && <NetworkInput pixels={selectedChar.nnPixels} />}
                  </span>
                  {selectedChar.nnCandidates ? (
                    <CandidateBars candidates={selectedChar.nnCandidates} />
                  ) : (
                    <div className="text-xs text-slate-400">Network not loaded</div>
                  )}
                </div>
                <div>
                  <span className="text-xs font-semibold text-slate-700 block mb-2">Hand-written rules (features above)</span>
                  <CandidateBars candidates={selectedChar.candidates} />
                </div>
              </div>
              <p className="text-[11px] text-slate-400">
                Note: EMNIST Balanced merges letters whose upper and lower case look alike (C/c, O/o, S/s, …), so the
                network answers with one class for both.
              </p>
            </div>
          ) : (
            <div className="py-12 text-center text-xs text-slate-400">
              Draw characters on the canvas above to inspect their topological features.
            </div>
          )}
        </div>

        {/* Right: Ambiguity Showdown & Comparison Lab */}
        <div className="lg:col-span-5 bg-white border border-slate-200 rounded-xl p-5 shadow-xs space-y-4">
          <div>
            <h3 className="text-sm font-semibold text-slate-900">Ambiguity Disambiguation Lab</h3>
            <p className="text-xs text-slate-500">
              How computer vision separates confusable digit/letter pairs
            </p>
          </div>

          {/* Pair tabs */}
          <div className="flex items-center gap-1 overflow-x-auto pb-1">
            {(Object.keys(AMBIGUOUS_PAIRS) as Array<keyof typeof AMBIGUOUS_PAIRS>).map((key) => {
              const pair = AMBIGUOUS_PAIRS[key];
              const isActive = activePairKey === key;
              return (
                <button
                  key={key}
                  onClick={() => setActivePairKey(key)}
                  className={`px-2.5 py-1 text-xs rounded-md font-mono transition-colors whitespace-nowrap ${
                    isActive
                      ? 'bg-slate-900 text-white'
                      : 'bg-slate-100 text-slate-600 hover:bg-slate-200'
                  }`}
                >
                  {pair.testA} vs {pair.testB}
                </button>
              );
            })}
          </div>

          {/* Active Pair Details */}
          <div className="p-4 bg-slate-50 rounded-lg border border-slate-100 space-y-3">
            <div className="flex items-center justify-between">
              <span className="text-xs font-semibold text-slate-800">
                {AMBIGUOUS_PAIRS[activePairKey].title}
              </span>
              <div className="flex gap-2">
                <button
                  onClick={() => handleLoadSampleString(AMBIGUOUS_PAIRS[activePairKey].testA)}
                  className="px-2 py-0.5 text-xs font-mono bg-white border border-slate-200 rounded hover:bg-slate-100"
                >
                  Test "{AMBIGUOUS_PAIRS[activePairKey].testA}"
                </button>
                <button
                  onClick={() => handleLoadSampleString(AMBIGUOUS_PAIRS[activePairKey].testB)}
                  className="px-2 py-0.5 text-xs font-mono bg-white border border-slate-200 rounded hover:bg-slate-100"
                >
                  Test "{AMBIGUOUS_PAIRS[activePairKey].testB}"
                </button>
              </div>
            </div>

            <p className="text-xs text-slate-600 leading-relaxed">
              {AMBIGUOUS_PAIRS[activePairKey].difference}
            </p>
          </div>

          <div className="text-[11px] text-slate-400">
            Tip: a network trained only on MNIST can only answer 0-9: it has 10 output neurons. To read letters the
            output layer must grow to 47 classes and the network must be trained on letters too.
          </div>
        </div>
      </div>

      <LettersTrainer onModelTrained={setModel} />
    </div>
  );
};

const CandidateBars: React.FC<{ candidates: Array<{ char: string; confidence: number }> }> = ({ candidates }) => (
  <div className="space-y-1.5">
    {candidates.map((cand, idx) => (
      <div key={cand.char} className="flex items-center gap-3 text-xs">
        <span className="w-5 font-mono font-bold text-slate-800 text-center">{cand.char}</span>
        <div className="flex-1 h-3 bg-slate-100 rounded-xs overflow-hidden">
          <div
            className={`h-full transition-all ${idx === 0 ? 'bg-slate-900' : 'bg-slate-300'}`}
            style={{ width: `${Math.round(cand.confidence * 100)}%` }}
          />
        </div>
        <span className="w-12 text-right font-mono text-slate-600 tabular-nums text-[11px]">
          {(cand.confidence * 100).toFixed(1)}%
        </span>
      </div>
    ))}
  </div>
);

const NetworkInput: React.FC<{ pixels: Float32Array }> = ({ pixels }) => {
  const ref = useRef<HTMLCanvasElement | null>(null);
  useEffect(() => {
    const ctx = ref.current?.getContext('2d');
    if (!ctx) return;
    const img = ctx.createImageData(28, 28);
    for (let i = 0; i < 784; i++) {
      const v = Math.round(255 * pixels[i]);
      img.data.set([v, v, v, 255], i * 4);
    }
    ctx.putImageData(img, 0, 0);
  }, [pixels]);
  return (
    <canvas
      ref={ref}
      width={28}
      height={28}
      title="28x28 input the network sees"
      className="w-7 h-7 rounded-xs bg-slate-900 [image-rendering:pixelated]"
    />
  );
};
