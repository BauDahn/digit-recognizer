import React, { useEffect, useRef, useState, useCallback } from 'react';
import { NeuralNetwork } from '../../ml/neuralNetwork';
import { createSyntheticDataset, generateSyntheticDigit } from '../../ml/syntheticDataset';
import { getRealMnistSplits, MnistSampleInfo } from '../../ml/mnistLoader';
import { DIGIT_CLASSES } from '../../ml/classes';
import {
  loadBuiltinModel,
  loadModelFromStorage,
  removeModelFromStorage,
  saveModelToStorage,
  toModelFile,
} from '../../ml/modelFile';
import {
  createTrainerWorker,
  getWeights,
  packSamples,
  setWeights,
  TrainRequest,
  WorkerMessage,
} from '../../ml/trainingClient';
import { preprocessCanvasTo28x28 } from '../../ml/preprocessing';
import { PredictionResult, TrainingEpochMetric, TrainingSample } from '../../types';
import { ConfusionMatrix } from './ConfusionMatrix';
import { WeightVisualizer } from './WeightVisualizer';
import { MnistGallery } from './MnistGallery';
import { TrainingCharts } from './TrainingCharts';
import { AugmentationPreview, LayerActivations } from './NetworkInsights';
import { LabInsights } from '../insights/LabInsights';
import { useWeightTracker } from '../insights/useWeightTracker';
import {
  Play,
  Pause,
  RotateCcw,
  Plus,
  Trash2,
  Database,
  Sparkles,
  CheckCircle2,
  Download,
  Save,
  Upload,
} from 'lucide-react';

interface DigitLabProps {
  network: NeuralNetwork;
  onNetworkUpdate: () => void;
}

type DatasetSource = 'real-mnist' | 'synthetic';

const STORAGE_KEY = 'digits';

export const DigitLab: React.FC<DigitLabProps> = ({ network, onNetworkUpdate }) => {
  // Drawing Canvas
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const previewCanvasRef = useRef<HTMLCanvasElement | null>(null);
  const [isDrawing, setIsDrawing] = useState(false);
  const [brushSize, setBrushSize] = useState<number>(18);
  const [prediction, setPrediction] = useState<PredictionResult | null>(null);
  const [activePixels, setActivePixels] = useState<Float32Array>(new Float32Array(784));

  // Training state
  const [isTraining, setIsTraining] = useState(false);
  const [currentEpoch, setCurrentEpoch] = useState(0);
  const [trainingMetrics, setTrainingMetrics] = useState<TrainingEpochMetric[]>([]);
  const [epochsTarget, setEpochsTarget] = useState<number>(10);
  const [learningRate, setLearningRate] = useState<number>(0.04);
  const [batchSize, setBatchSize] = useState<number>(16);
  const [useAugmentation, setUseAugmentation] = useState(true);
  const workerRef = useRef<Worker | null>(null);

  // Dataset selection: 'real-mnist' vs 'synthetic'
  const [datasetSource, setDatasetSource] = useState<DatasetSource>('real-mnist');
  const [dataset, setDataset] = useState<TrainingSample[]>([]);
  const [testDataset, setTestDataset] = useState<TrainingSample[]>([]);
  const [datasetStatus, setDatasetStatus] = useState<'loading' | 'ready' | 'error'>('loading');
  const [confusionMatrix, setConfusionMatrix] = useState<number[][]>(() =>
    Array.from({ length: 10 }, () => Array(10).fill(0))
  );
  const [testAccuracy, setTestAccuracy] = useState<number | null>(null);

  // Custom sample label adding
  const tracker = useWeightTracker(network);
  const [customLabel, setCustomLabel] = useState<number>(0);
  const [addedToast, setAddedToast] = useState<string | null>(null);
  const [modelStatus, setModelStatus] = useState<string>('Untrained network (random weights)');

  const showToast = useCallback((text: string) => {
    setAddedToast(text);
    setTimeout(() => setAddedToast(null), 3000);
  }, []);

  const refreshEvaluation = useCallback(
    (testSet: TrainingSample[]) => {
      if (testSet.length === 0) return;
      const evalRes = network.evaluate(testSet);
      setConfusionMatrix(evalRes.confusionMatrix);
      setTestAccuracy(evalRes.accuracy);
    },
    [network]
  );

  // Restore the model saved in this browser (once per session)
  useEffect(() => {
    if (network.isTrained) {
      setModelStatus('Model trained in this session');
      return;
    }
    const saved = loadModelFromStorage(STORAGE_KEY);
    if (!saved) return;
    try {
      network.importWeights(JSON.stringify(saved));
      setModelStatus(`Restored your saved model${saved.meta?.createdAt ? ` (${saved.meta.createdAt})` : ''}`);
      onNetworkUpdate();
    } catch {
      removeModelFromStorage(STORAGE_KEY);
    }
  }, [network, onNetworkUpdate]);

  // Metrics always describe the weights the network has now (a model may be loaded before or after the test set)
  useEffect(() => {
    if (!isTraining && testDataset.length > 0) refreshEvaluation(testDataset);
  }, [testDataset, modelStatus, isTraining, refreshEvaluation]);

  // The weights at this moment are where the weight-change tracking starts from
  useEffect(() => {
    tracker.reset();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // Load dataset based on selected source; custom samples the user taught are kept
  const loadDataset = useCallback(
    async (source: DatasetSource) => {
      setDatasetStatus('loading');
      try {
        const { trainSet, testSet } =
          source === 'real-mnist'
            ? await getRealMnistSplits()
            : { trainSet: createSyntheticDataset(30), testSet: createSyntheticDataset(10) };
        setDataset((prev) => [...prev.filter((s) => s.isCustom), ...trainSet]);
        setTestDataset(testSet);
        refreshEvaluation(testSet);
        setDatasetStatus('ready');
      } catch (e) {
        console.error(e);
        setDatasetStatus('error');
      }
    },
    [refreshEvaluation]
  );

  useEffect(() => {
    loadDataset(datasetSource);
  }, [datasetSource, loadDataset]);

  useEffect(() => () => workerRef.current?.terminate(), []);

  // Setup drawing canvas
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    // Fill white background
    ctx.fillStyle = '#ffffff';
    ctx.fillRect(0, 0, canvas.width, canvas.height);
  }, []);

  // Update 28x28 normalized preview canvas
  const updatePreview = useCallback(
    (pixels: Float32Array) => {
      const pCanvas = previewCanvasRef.current;
      if (!pCanvas) return;
      const pctx = pCanvas.getContext('2d');
      if (!pctx) return;

      const imgData = pctx.createImageData(28, 28);
      for (let i = 0; i < 784; i++) {
        // High activation (ink) is black, background is white
        const val = Math.round(255 * (1 - pixels[i]));
        imgData.data[i * 4] = val;
        imgData.data[i * 4 + 1] = val;
        imgData.data[i * 4 + 2] = val;
        imgData.data[i * 4 + 3] = 255;
      }
      pctx.putImageData(imgData, 0, 0);
    },
    []
  );

  // Run prediction on current drawing
  const classifyCurrentCanvas = useCallback(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const { pixels, boundingBox } = preprocessCanvasTo28x28(canvas);
    setActivePixels(pixels);
    updatePreview(pixels);

    if (boundingBox) {
      const pred = network.predict(pixels);
      setPrediction(pred);
    } else {
      setPrediction(null);
    }
  }, [network, updatePreview]);

  // Canvas drawing handlers
  const handlePointerDown = (e: React.PointerEvent<HTMLCanvasElement>) => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const rect = canvas.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;

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
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;

    ctx.lineTo(x, y);
    ctx.stroke();
    classifyCurrentCanvas();
  };

  const handlePointerUp = (e: React.PointerEvent<HTMLCanvasElement>) => {
    if (!isDrawing) return;
    setIsDrawing(false);
    classifyCurrentCanvas();
    try {
      e.currentTarget.releasePointerCapture(e.pointerId);
    } catch {
      // ignore
    }
  };

  const handleClearCanvas = () => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    ctx.fillStyle = '#ffffff';
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    setPrediction(null);
    const emptyPixels = new Float32Array(784);
    setActivePixels(emptyPixels);
    updatePreview(emptyPixels);
  };

  // Add current drawn digit to training dataset
  const handleAddToDataset = () => {
    if (!prediction) {
      showToast('Draw a digit first, then choose its label and click "Teach Model".');
      return;
    }

    const newSample: TrainingSample = {
      id: `custom-${Date.now()}`,
      label: customLabel,
      pixels: new Float32Array(activePixels),
      isCustom: true,
    };

    setDataset((prev) => [newSample, ...prev]);
    showToast(`Added your drawing as digit "${customLabel}". Train again to learn from it.`);
  };

  // Load authentic MNIST sample onto canvas
  const handleSelectMnistSample = (sample: MnistSampleInfo) => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    ctx.fillStyle = '#ffffff';
    ctx.fillRect(0, 0, canvas.width, canvas.height);

    // Render 28x28 sample scaled up to 280x280
    const tempCanvas = document.createElement('canvas');
    tempCanvas.width = 28;
    tempCanvas.height = 28;
    const tctx = tempCanvas.getContext('2d')!;
    const tImg = tctx.createImageData(28, 28);

    for (let i = 0; i < 784; i++) {
      // High activation is black ink
      const v = Math.round(255 * (1 - sample.pixels[i]));
      tImg.data[i * 4] = v;
      tImg.data[i * 4 + 1] = v;
      tImg.data[i * 4 + 2] = v;
      tImg.data[i * 4 + 3] = 255;
    }
    tctx.putImageData(tImg, 0, 0);

    ctx.imageSmoothingEnabled = false;
    ctx.drawImage(tempCanvas, 0, 0, canvas.width, canvas.height);

    classifyCurrentCanvas();
    showToast(`Loaded MNIST test sample (true label: ${sample.label})`);
  };

  // Load sample digit onto canvas
  const handleLoadSample = (digit: number) => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    ctx.fillStyle = '#ffffff';
    ctx.fillRect(0, 0, canvas.width, canvas.height);

    const synthPixels = generateSyntheticDigit(digit, { lineWidth: 3.5, noise: 0 });
    // Scale 28x28 up to 280x280
    const tempCanvas = document.createElement('canvas');
    tempCanvas.width = 28;
    tempCanvas.height = 28;
    const tctx = tempCanvas.getContext('2d')!;
    const tImg = tctx.createImageData(28, 28);
    for (let i = 0; i < 784; i++) {
      const v = Math.round(255 * (1 - synthPixels[i]));
      tImg.data[i * 4] = v;
      tImg.data[i * 4 + 1] = v;
      tImg.data[i * 4 + 2] = v;
      tImg.data[i * 4 + 3] = 255;
    }
    tctx.putImageData(tImg, 0, 0);

    ctx.imageSmoothingEnabled = false;
    ctx.drawImage(tempCanvas, 0, 0, canvas.width, canvas.height);

    classifyCurrentCanvas();
  };

  const persistModel = useCallback(
    (accuracy: number | null, epochs: number) => {
      const file = toModelFile('digits', DIGIT_CLASSES, network, {
        trainedOn: datasetSource === 'real-mnist' ? 'MNIST subset (browser)' : 'Synthetic digits (browser)',
        testAccuracy: accuracy ?? undefined,
        epochs,
        createdAt: new Date().toISOString().slice(0, 10),
      });
      const ok = saveModelToStorage(STORAGE_KEY, file);
      setModelStatus(ok ? 'Model trained here and saved in this browser' : 'Model trained here (could not save it)');
    },
    [datasetSource, network]
  );

  // Training runs in a Web Worker: the UI keeps responding while the network learns
  const handleStartTraining = () => {
    if (dataset.length === 0 || testDataset.length === 0) return;
    workerRef.current?.terminate();
    const worker = createTrainerWorker();
    workerRef.current = worker;

    const startEpoch = trainingMetrics.length > 0 ? trainingMetrics[trainingMetrics.length - 1].epoch : 0;
    tracker.rebase();
    setIsTraining(true);

    worker.onmessage = (event: MessageEvent<WorkerMessage>) => {
      const msg = event.data;
      if (msg.type === 'progress') {
        // weights in the middle of an epoch: let the diagram show the neurons being adjusted
        setWeights(network, msg.weights);
        tracker.recordUpdate(startEpoch + msg.epoch - 1 + msg.fraction);
        classifyCurrentCanvas();
        onNetworkUpdate();
      } else if (msg.type === 'epoch') {
        setWeights(network, msg.weights);
        const epoch = startEpoch + msg.epoch;
        tracker.recordUpdate(epoch);
        setCurrentEpoch(epoch);
        setConfusionMatrix(msg.confusionMatrix);
        setTestAccuracy(msg.testAccuracy);
        setTrainingMetrics((prev) => [
          ...prev,
          { epoch, loss: msg.loss, accuracy: msg.testAccuracy, trainAccuracy: msg.trainAccuracy, seconds: msg.seconds },
        ]);
        persistModel(msg.testAccuracy, epoch);
        classifyCurrentCanvas();
        onNetworkUpdate();
      } else {
        if (msg.type === 'error') showToast(`Training failed: ${msg.message}`);
        worker.terminate();
        workerRef.current = null;
        setIsTraining(false);
      }
    };

    const request: TrainRequest = {
      type: 'train',
      config: { ...network.config, learningRate, batchSize },
      initialWeights: network.isTrained ? getWeights(network) : null,
      train: packSamples(dataset),
      test: packSamples(testDataset),
      epochs: epochsTarget,
      augmentProbability: useAugmentation ? 0.5 : 0,
      updatesPerEpoch: 5,
    };
    worker.postMessage(request);
  };

  // Stopping = terminating the worker. The weights of the last finished epoch are kept.
  const handleStopTraining = () => {
    workerRef.current?.terminate();
    workerRef.current = null;
    setIsTraining(false);
  };

  const handleResetWeights = () => {
    handleStopTraining();
    network.initWeightsHe();
    tracker.reset();
    setCurrentEpoch(0);
    setTrainingMetrics([]);
    removeModelFromStorage(STORAGE_KEY);
    setModelStatus('Untrained network (random weights)');
    refreshEvaluation(testDataset);
    classifyCurrentCanvas();
    onNetworkUpdate();
  };

  const handleLoadPretrained = async () => {
    try {
      const model = await loadBuiltinModel('digits');
      network.copyWeightsFrom(model.network);
      tracker.reset();
      setTrainingMetrics([]);
      setCurrentEpoch(0);
      const acc = model.meta?.testAccuracy;
      setModelStatus(
        `Pre-trained on ${model.meta?.trainedOn ?? 'MNIST'}${acc ? ` (${(acc * 100).toFixed(1)}% on the full test set)` : ''}`
      );
      refreshEvaluation(testDataset);
      classifyCurrentCanvas();
      onNetworkUpdate();
    } catch (e: any) {
      showToast(`Could not load the pre-trained model: ${e?.message || e}`);
    }
  };

  const handleExportModel = () => {
    const file = toModelFile('digits', DIGIT_CLASSES, network, {
      testAccuracy: testAccuracy ?? undefined,
      createdAt: new Date().toISOString().slice(0, 10),
    });
    const blob = new Blob([JSON.stringify(file)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url;
    link.download = 'neuroscribe-digits-model.json';
    link.click();
    URL.revokeObjectURL(url);
  };

  const importInputRef = useRef<HTMLInputElement | null>(null);
  const handleImportModel = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const fileInput = e.target.files?.[0];
    e.target.value = '';
    if (!fileInput) return;
    try {
      network.importWeights(await fileInput.text());
      tracker.reset();
      setTrainingMetrics([]);
      setCurrentEpoch(0);
      persistModel(null, 0);
      setModelStatus(`Imported ${fileInput.name}`);
      refreshEvaluation(testDataset);
      classifyCurrentCanvas();
      onNetworkUpdate();
    } catch (err: any) {
      showToast(`Import failed: ${err?.message || err}`);
    }
  };

  return (
    <div className="max-w-7xl mx-auto px-4 sm:px-6 py-6 space-y-6">
      {/* Top Banner / Explanation */}
      <div className="flex flex-col md:flex-row md:items-center justify-between gap-4 pb-4 border-b border-slate-200">
        <div>
          <h1 className="text-xl font-bold text-slate-900 tracking-tight">
            Digit Neural Lab
          </h1>
          <p className="text-xs text-slate-500 mt-1 max-w-2xl">
            Train a real neural network (Multi-Layer Perceptron) with Backpropagation directly inside your browser. Draw digits, inspect the 28×28 center-of-mass pre-processing, observe live activation filters, and analyze the 10×10 confusion matrix.
          </p>
        </div>

        {/* Quick sample digit pills */}
        <div className="flex items-center gap-1.5 flex-wrap">
          <span className="text-xs text-slate-500 font-medium mr-1">Load Sample:</span>
          {[0, 1, 2, 3, 4, 5, 6, 7, 8, 9].map((d) => (
            <button
              key={d}
              onClick={() => handleLoadSample(d)}
              className="w-7 h-7 text-xs font-mono font-medium rounded-md bg-white border border-slate-200 hover:bg-slate-100 text-slate-700 transition-colors shadow-2xs"
            >
              {d}
            </button>
          ))}
        </div>
      </div>

      {/* Main Grid: Interactive Canvas + Realtime Predictions + Training Controller */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
        {/* Left Column: Drawing Pad & Preprocessing (5 cols) */}
        <div className="lg:col-span-5 space-y-4">
          <div className="bg-white border border-slate-200 rounded-xl p-5 shadow-xs">
            <div className="flex items-center justify-between mb-3">
              <span className="text-xs font-semibold uppercase tracking-wider text-slate-500">
                Interactive Drawing Pad
              </span>
              <div className="flex items-center gap-2">
                <label className="text-[11px] text-slate-500 flex items-center gap-1">
                  Brush:
                  <input
                    type="range"
                    min="10"
                    max="28"
                    value={brushSize}
                    onChange={(e) => setBrushSize(Number(e.target.value))}
                    className="w-16 h-1 bg-slate-200 rounded-lg appearance-none cursor-pointer"
                  />
                </label>
                <button
                  onClick={handleClearCanvas}
                  className="p-1.5 text-slate-500 hover:text-rose-600 rounded-md hover:bg-slate-100 transition-colors"
                  title="Clear drawing"
                >
                  <Trash2 className="w-4 h-4" />
                </button>
              </div>
            </div>

            {/* Drawing Canvas */}
            <div className="relative flex justify-center bg-slate-50 rounded-lg p-2 border border-slate-200">
              <canvas
                ref={canvasRef}
                width={280}
                height={280}
                onPointerDown={handlePointerDown}
                onPointerMove={handlePointerMove}
                onPointerUp={handlePointerUp}
                className="w-[280px] h-[280px] bg-white rounded-md shadow-xs cursor-crosshair touch-none border border-slate-300"
              />
            </div>

            {/* Normalized 28x28 Preprocessing View & Add to Dataset */}
            <div className="mt-4 pt-4 border-t border-slate-100 flex items-center justify-between">
              <div className="flex items-center gap-3">
                <div className="p-1 bg-slate-900 rounded-md">
                  <canvas
                    ref={previewCanvasRef}
                    width={28}
                    height={28}
                    className="w-10 h-10 image-rendering-pixelated rounded-xs"
                    title="28x28 Center-of-Mass Centered MNIST input"
                  />
                </div>
                <div>
                  <div className="text-xs font-semibold text-slate-800">28×28 Pre-processed</div>
                  <div className="text-[11px] text-slate-400">Centered by Center-of-Mass</div>
                </div>
              </div>

              {/* Add to Dataset controls */}
              <div className="flex items-center gap-2">
                <select
                  value={customLabel}
                  onChange={(e) => setCustomLabel(Number(e.target.value))}
                  className="px-2 py-1 text-xs font-mono bg-white border border-slate-200 rounded-md focus:outline-hidden"
                >
                  {[0, 1, 2, 3, 4, 5, 6, 7, 8, 9].map((num) => (
                    <option key={num} value={num}>
                      Digit {num}
                    </option>
                  ))}
                </select>
                <button
                  onClick={handleAddToDataset}
                  className="flex items-center gap-1 px-2.5 py-1 text-xs font-medium text-slate-700 bg-slate-100 hover:bg-slate-200 rounded-md transition-colors"
                  title="Add this drawing to your training set"
                >
                  <Plus className="w-3.5 h-3.5" />
                  <span>Teach Model</span>
                </button>
              </div>
            </div>

            {addedToast && (
              <div className="mt-2 text-xs text-emerald-700 bg-emerald-50 px-3 py-1.5 rounded-md flex items-center gap-1.5 animate-in fade-in">
                <CheckCircle2 className="w-3.5 h-3.5 text-emerald-600 shrink-0" />
                <span>{addedToast}</span>
              </div>
            )}
          </div>
        </div>

        {/* Middle Column: Prediction Probabilities (3 cols) */}
        <div className="lg:col-span-3 space-y-4">
          <div className="bg-white border border-slate-200 rounded-xl p-5 shadow-xs h-full flex flex-col justify-between">
            <div>
              <div className="flex items-center justify-between mb-3">
                <span className="text-xs font-semibold uppercase tracking-wider text-slate-500">
                  Model Prediction
                </span>
                {prediction && (
                  <span className="text-xs font-mono text-emerald-600 font-semibold">
                    {(prediction.confidence * 100).toFixed(1)}%
                  </span>
                )}
              </div>

              {/* Top Banner */}
              <div className="p-4 bg-slate-50 rounded-lg text-center border border-slate-100 mb-4">
                <div className="text-[11px] text-slate-400 uppercase tracking-wider">
                  Recognized Digit
                </div>
                <div className="text-5xl font-mono font-bold text-slate-900 mt-1">
                  {prediction !== null ? prediction.digit : '—'}
                </div>
                <div className="text-xs text-slate-500 mt-1">
                  {prediction !== null
                    ? `Confidence: ${(prediction.confidence * 100).toFixed(1)}%`
                    : 'Draw on the pad to test'}
                </div>
              </div>

              {/* 0-9 Softmax Probability Distribution */}
              <div className="space-y-1.5">
                {Array.from({ length: 10 }, (_, d) => {
                  const prob = prediction ? prediction.probabilities[d] || 0 : 0;
                  const isTop = prediction?.digit === d;
                  return (
                    <div key={d} className="flex items-center gap-2 text-xs">
                      <span
                        className={`w-4 font-mono font-semibold ${
                          isTop ? 'text-slate-900' : 'text-slate-400'
                        }`}
                      >
                        {d}
                      </span>
                      <div className="flex-1 h-3.5 bg-slate-100 rounded-xs overflow-hidden relative">
                        <div
                          className={`h-full transition-all duration-150 ${
                            isTop ? 'bg-slate-900' : 'bg-slate-300'
                          }`}
                          style={{ width: `${Math.round(prob * 100)}%` }}
                        />
                      </div>
                      <span className="w-10 text-right font-mono text-[11px] text-slate-500 tabular-nums">
                        {(prob * 100).toFixed(0)}%
                      </span>
                    </div>
                  );
                })}
              </div>
            </div>

            <div className="pt-3 border-t border-slate-100 text-[11px] text-slate-400">
              Softmax output layer with Cross-Entropy loss
            </div>
          </div>
        </div>

        {/* Right Column: Training Controls & Live Telemetry (4 cols) */}
        <div className="lg:col-span-4 space-y-4">
          <div className="bg-white border border-slate-200 rounded-xl p-5 shadow-xs">
            <div className="flex items-center justify-between mb-3">
              <span className="text-xs font-semibold uppercase tracking-wider text-slate-500">
                Neural Training Engine
              </span>
              <span className="text-xs text-slate-500 font-mono">
                {datasetStatus === 'loading'
                  ? 'loading…'
                  : `${dataset.length} train · ${testDataset.length} test`}
              </span>
            </div>

            {/* Current model + persistence */}
            <div className="mb-4 p-3 bg-slate-50 rounded-lg border border-slate-100 space-y-2">
              <div className="text-xs text-slate-700">
                <span className="font-semibold">Model: </span>
                {modelStatus}
              </div>
              <div className="flex flex-wrap gap-1.5">
                <button
                  onClick={handleLoadPretrained}
                  disabled={isTraining}
                  className="flex items-center gap-1 px-2 py-1 text-[11px] font-medium bg-white border border-slate-200 rounded-md hover:bg-slate-100 disabled:opacity-50"
                  title="Load weights trained on the full MNIST dataset (60,000 images)"
                >
                  <Download className="w-3 h-3" /> Pre-trained
                </button>
                <button
                  onClick={handleExportModel}
                  className="flex items-center gap-1 px-2 py-1 text-[11px] font-medium bg-white border border-slate-200 rounded-md hover:bg-slate-100"
                  title="Download the current weights as a .json file"
                >
                  <Save className="w-3 h-3" /> Export
                </button>
                <button
                  onClick={() => importInputRef.current?.click()}
                  disabled={isTraining}
                  className="flex items-center gap-1 px-2 py-1 text-[11px] font-medium bg-white border border-slate-200 rounded-md hover:bg-slate-100 disabled:opacity-50"
                  title="Load weights from a .json file exported before"
                >
                  <Upload className="w-3 h-3" /> Import
                </button>
                <input ref={importInputRef} type="file" accept="application/json,.json" className="hidden" onChange={handleImportModel} />
              </div>
            </div>

            {datasetStatus === 'error' && (
              <div className="mb-3 text-xs text-rose-700 bg-rose-50 px-3 py-1.5 rounded-md">
                Could not load the MNIST files from /datasets. Run <code>npm run build:datasets</code>.
              </div>
            )}

            {/* Hyperparameters */}
            <div className="space-y-3 mb-4 text-xs">
              {/* Dataset Source Selector */}
              <div>
                <span className="text-slate-600 block mb-1 font-medium">Training Dataset Source:</span>
                <div className="grid grid-cols-2 gap-1 p-0.5 bg-slate-100 rounded-lg">
                  <button
                    onClick={() => setDatasetSource('real-mnist')}
                    disabled={isTraining}
                    className={`py-1.5 px-2 text-xs font-medium rounded-md transition-colors flex items-center justify-center gap-1.5 ${
                      datasetSource === 'real-mnist'
                        ? 'bg-white text-indigo-700 shadow-2xs font-semibold'
                        : 'text-slate-600 hover:text-slate-900'
                    }`}
                  >
                    <Database className="w-3 h-3 text-indigo-500" />
                    <span>Real MNIST</span>
                  </button>
                  <button
                    onClick={() => setDatasetSource('synthetic')}
                    disabled={isTraining}
                    className={`py-1.5 px-2 text-xs font-medium rounded-md transition-colors flex items-center justify-center gap-1.5 ${
                      datasetSource === 'synthetic'
                        ? 'bg-white text-slate-900 shadow-2xs font-semibold'
                        : 'text-slate-600 hover:text-slate-900'
                    }`}
                  >
                    <Sparkles className="w-3 h-3 text-amber-500" />
                    <span>Synthetic</span>
                  </button>
                </div>
              </div>

              <div>
                <div className="flex justify-between text-slate-600 mb-1">
                  <span>Epochs</span>
                  <span className="font-mono font-semibold">{epochsTarget}</span>
                </div>
                <input
                  type="range"
                  min="1"
                  max="30"
                  step="1"
                  value={epochsTarget}
                  onChange={(e) => setEpochsTarget(Number(e.target.value))}
                  disabled={isTraining}
                  className="w-full h-1.5 bg-slate-200 rounded-lg appearance-none cursor-pointer"
                />
              </div>

              <div>
                <div className="flex justify-between text-slate-600 mb-1">
                  <span>Learning Rate (α)</span>
                  <span className="font-mono font-semibold">{learningRate}</span>
                </div>
                <input
                  type="range"
                  min="0.01"
                  max="0.15"
                  step="0.01"
                  value={learningRate}
                  onChange={(e) => setLearningRate(Number(e.target.value))}
                  disabled={isTraining}
                  className="w-full h-1.5 bg-slate-200 rounded-lg appearance-none cursor-pointer"
                />
              </div>

              <label className="flex items-center justify-between cursor-pointer">
                <span className="text-slate-600">Data augmentation</span>
                <input
                  type="checkbox"
                  checked={useAugmentation}
                  onChange={(e) => setUseAugmentation(e.target.checked)}
                  disabled={isTraining}
                  className="accent-slate-900"
                />
              </label>

              <div className="flex items-center justify-between">
                <span className="text-slate-600">Batch Size</span>
                <div className="flex gap-1">
                  {[8, 16, 32].map((bs) => (
                    <button
                      key={bs}
                      onClick={() => setBatchSize(bs)}
                      disabled={isTraining}
                      className={`px-2 py-0.5 text-xs font-mono rounded ${
                        batchSize === bs
                          ? 'bg-slate-900 text-white'
                          : 'bg-slate-100 text-slate-600 hover:bg-slate-200'
                      }`}
                    >
                      {bs}
                    </button>
                  ))}
                </div>
              </div>
            </div>

            {/* Training Action Buttons */}
            <div className="flex items-center gap-2 mb-4">
              {!isTraining ? (
                <button
                  onClick={handleStartTraining}
                  disabled={datasetStatus !== 'ready'}
                  className="flex-1 flex items-center justify-center gap-1.5 py-2 px-3 bg-slate-900 text-white rounded-lg text-xs font-medium hover:bg-slate-800 transition-colors shadow-xs"
                >
                  <Play className="w-3.5 h-3.5 fill-current" />
                  <span>{trainingMetrics.length > 0 || network.isTrained ? 'Continue Training' : 'Train Model'}</span>
                </button>
              ) : (
                <button
                  onClick={handleStopTraining}
                  className="flex-1 flex items-center justify-center gap-1.5 py-2 px-3 bg-amber-600 text-white rounded-lg text-xs font-medium hover:bg-amber-700 transition-colors shadow-xs"
                >
                  <Pause className="w-3.5 h-3.5 fill-current" />
                  <span>Stop (keeps last epoch)</span>
                </button>
              )}

              <button
                onClick={handleResetWeights}
                disabled={isTraining}
                className="p-2 text-slate-600 hover:text-slate-900 rounded-lg border border-slate-200 hover:bg-slate-50 transition-colors"
                title="Reset weights (He initialization)"
              >
                <RotateCcw className="w-4 h-4" />
              </button>
            </div>

            {/* Live Progress / Metrics */}
            <div className="bg-slate-50 rounded-lg p-3 border border-slate-100 space-y-2">
              <div className="flex justify-between text-xs">
                <span className="text-slate-500">Epoch:</span>
                <span className="font-mono font-semibold text-slate-800">
                  {currentEpoch}
                  {isTraining && trainingMetrics.length > 0 && trainingMetrics[trainingMetrics.length - 1].seconds !== undefined
                    ? ` (${trainingMetrics[trainingMetrics.length - 1].seconds!.toFixed(1)}s / epoch)`
                    : ''}
                </span>
              </div>
              <div className="flex justify-between text-xs">
                <span className="text-slate-500">Validation Accuracy:</span>
                <span className="font-mono font-semibold text-emerald-600">
                  {testAccuracy !== null ? `${(testAccuracy * 100).toFixed(1)}%` : '—'}
                </span>
              </div>
              <div className="flex justify-between text-xs">
                <span className="text-slate-500">Current Loss:</span>
                <span className="font-mono text-slate-600">
                  {trainingMetrics.length > 0
                    ? trainingMetrics[trainingMetrics.length - 1].loss.toFixed(4)
                    : '—'}
                </span>
              </div>
            </div>

            <TrainingCharts metrics={trainingMetrics} />
          </div>
        </div>
      </div>

      {/* Inside the network: graph, parameters, weight dynamics and metrics */}
      <LabInsights
        network={network}
        classes={DIGIT_CLASSES}
        pixels={activePixels}
        tracker={tracker}
        confusion={confusionMatrix}
        training={isTraining}
        hyper={{ learningRate, momentum: network.config.momentum, batchSize, epochs: epochsTarget, augmentation: useAugmentation }}
      />

      {/* What happens inside: activations per layer + data augmentation */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
        <div className="lg:col-span-7">
          <LayerActivations network={network} pixels={activePixels} />
        </div>
        <div className="lg:col-span-5">
          <AugmentationPreview pixels={activePixels} enabled={useAugmentation} />
        </div>
      </div>

      {/* Real MNIST Dataset Visual Browser & Explorer */}
      <MnistGallery onSelectSample={handleSelectMnistSample} />

      {/* Lower Row: Confusion Matrix + Neural Feature Weight Visualizer */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
        <div className="lg:col-span-6">
          <ConfusionMatrix matrix={confusionMatrix} />
        </div>
        <div className="lg:col-span-6">
          <WeightVisualizer
            weightsW1={network.w1}
            hiddenSize1={network.config.hiddenSize1}
            activations={prediction?.rawActivations}
          />
        </div>
      </div>
    </div>
  );
};
